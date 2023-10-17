from djl_python.huggingface import HuggingFaceService
from djl_python import Output
from djl_python.encode_decode import encode, decode
import logging
import json
import types

_service = HuggingFaceService()


def custom_parse_input(self, inputs):
    input_data = []
    input_size = []
    parameters = []
    errors = {}
    batch = inputs.get_batches()
    for i, item in enumerate(batch):
        try:
            content_type = item.get_property("Content-Type")
            input_map = decode(item, content_type)

            logging.info(f"input_map is #{json.dumps(input_map)}")
            _inputs = input_map.pop("prompt", input_map)
            logging.info(f"input_map after pop is #{json.dumps(input_map)}")

            parameters.append(input_map)
            for i in parameters:
                logging.info(f"parameters is #{json.dumps(i)}")

            # parameters.append(input_map.pop("parameters", {}))
            if isinstance(_inputs, list):
                input_data.extend(_inputs)
                input_size.append(len(_inputs))
            else:
                input_data.append(_inputs)
                input_size.append(1)

        except Exception as e:  # pylint: disable=broad-except
            logging.exception(f"Parse input failed: {i}")
            errors[i] = str(e)
    logging.info(f"returned values #{input_data}, #{input_size}, #{parameters}, #{errors}, #{batch} ")
    return input_data, input_size, parameters, errors, batch


def handle(inputs):
    if not _service.initialized:
        _service.initialize(inputs.get_properties())
        # replace parse_input
        _service.parse_input = types.MethodType(custom_parse_input, _service)

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
