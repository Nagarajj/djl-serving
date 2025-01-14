/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.python.engine;

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.metric.Metric;
import ai.djl.metric.Unit;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;
import ai.djl.util.RandomUtils;

import com.google.gson.JsonObject;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

class RollingBatch implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(RollingBatch.class);
    private static final Logger SERVER_METRIC = LoggerFactory.getLogger("server_metric");

    private static ExecutorService threadPool = Executors.newCachedThreadPool();

    private PyProcess process;
    private int maxRollingBatchSize;
    private int timeout;
    private String contentType;
    private boolean stop;
    private List<Request> list;
    private Thread currentThread;
    private ReentrantLock lock;
    private Condition canAdd;
    private Condition canRead;
    private boolean resetRollingBatch;

    RollingBatch(PyProcess process, int maxRollingBatchSize, int timeout, String outputFormatter) {
        this.process = process;
        this.maxRollingBatchSize = maxRollingBatchSize;
        this.timeout = timeout;
        if (outputFormatter == null || "json".equals(outputFormatter)) {
            contentType = "application/json";
        } else if ("jsonlines".equals(outputFormatter)) {
            contentType = "application/jsonlines";
        }
        // TODO: find a way to support custom output_formatter
        list = new ArrayList<>(3);
        lock = new ReentrantLock(true);
        canAdd = lock.newCondition();
        canRead = lock.newCondition();
        threadPool.submit(this);
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        currentThread = Thread.currentThread();
        while (!stop) {
            int size;
            Input batch = new Input();
            try {
                lock.lock();
                if (list.isEmpty()) {
                    canRead.await();
                }

                if (resetRollingBatch) {
                    batch.addProperty("reset_rollingbatch", "true");
                    resetRollingBatch = false;
                }
                size = list.size();
                for (int i = 0; i < size; ++i) {
                    Request req = list.get(i);
                    String prefix = "batch_" + i + '.';
                    for (Map.Entry<String, String> entry : req.input.getProperties().entrySet()) {
                        String key = prefix + entry.getKey();
                        batch.addProperty(key, entry.getValue());
                    }

                    batch.add(prefix + "data", req.getRequest());
                    String seed = req.getSeed();
                    if (seed != null) {
                        batch.add(prefix + "seed", req.seed);
                    }
                }
                batch.addProperty("batch_size", String.valueOf(size));
            } catch (InterruptedException e) {
                break;
            } finally {
                lock.unlock();
            }

            Output output = process.predict(batch, timeout, false);

            try {
                lock.lock();
                PairList<String, BytesSupplier> content = output.getContent();
                // TODO: optimize for conditional killing
                int code = output.getCode();
                if (code != 200 || content.size() != size) {
                    if (code != 200) {
                        logger.warn("Batch inference failed: {}", output.getMessage());
                    } else {
                        logger.error(
                                "Batch output size mismatch, expected: {}, actual: {}",
                                size,
                                content.size());
                    }
                    Output out = new Output(output.getCode(), "Batch inference failed");
                    BytesSupplier err = BytesSupplier.wrap(JsonUtils.GSON.toJson(out));
                    for (Request req : list) {
                        req.last = true;
                        req.data.appendContent(err, true);
                    }
                    list.clear();
                    resetRollingBatch = true;
                    canAdd.signal();
                    continue;
                }
                for (int i = 0; i < size; ++i) {
                    Request status = list.get(i);
                    String json = content.get(i).getValue().getAsString();
                    status.addResponse(json);
                }
                list.removeIf(status -> status.last);
                if (list.size() < maxRollingBatchSize) {
                    canAdd.signal();
                }
            } finally {
                lock.unlock();
            }
        }
    }

    public Output addInput(Input input, int timeout) throws TranslateException {
        try {
            lock.lock();
            if (list.size() >= maxRollingBatchSize) {
                logger.debug("exceed max_rolling_batch_size: {}", maxRollingBatchSize);
                if (!canAdd.await(timeout, TimeUnit.SECONDS)) {
                    SERVER_METRIC.info("{}", new Metric("RollingTimeout", list.size(), Unit.COUNT));
                    throw new TranslateException("Time out in: " + timeout);
                }
            }
            Request req = new Request(input, String.valueOf(RandomUtils.nextInt()), contentType);
            list.add(req);
            SERVER_METRIC.info("{}", new Metric("RollingBatchSize", list.size(), Unit.COUNT));
            canRead.signal();
            return req.output;
        } catch (InterruptedException e) {
            throw new TranslateException("Interrupted", e);
        } finally {
            lock.unlock();
        }
    }

    public void shutdown() {
        this.stop = true;
        threadPool.shutdown();
        currentThread.interrupt();
    }

    private static final class Request {

        Input input;
        ChunkedBytesSupplier data;
        Output output;
        String nextToken;
        boolean last;
        String seed;

        Request(Input input, String seed, String contentType) {
            this.input = input;
            data = new ChunkedBytesSupplier();
            output = new Output();
            if (contentType != null) {
                output.addProperty("Content-Type", contentType);
            }
            output.add(data);
            this.seed = seed;
        }

        BytesSupplier getRequest() {
            if (nextToken != null) {
                return BytesSupplier.wrap("{\"inputs\": [\"\"]}");
            }
            return input.getData();
        }

        /**
         * Seed is required for LMI Dist for sampling for all processes in the MPI to generate the
         * same token. NextTokenChooserParameters is constructed during first forward and preserved
         * for all forward calls of the request.
         *
         * @return seed, only for first forward
         */
        String getSeed() {
            if (nextToken != null) {
                return null;
            }
            return seed;
        }

        void addResponse(String json) {
            JsonObject element = JsonUtils.GSON.fromJson(json, JsonObject.class);
            last = element.get("last").getAsBoolean();
            nextToken = element.get("data").getAsString();
            try {
                JsonObject content = JsonUtils.GSON.fromJson(nextToken, JsonObject.class);
                output.setCode(content.get("code").getAsInt());
                output.setMessage(content.get("error").getAsString());
            } catch (Throwable ignore) {
                // ignore
            }

            data.appendContent(BytesSupplier.wrap(nextToken), last);
        }
    }
}
