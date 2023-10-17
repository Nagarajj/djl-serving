from unittest import TestCase

import test_model
from djl_python import Input

input_ = '''
'{
"input": {
    "prompt": "Bought book on human rights in usa with my card",
    "top_p": 1,
    "temperature": 0.7,
    "max_tokens": 100,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "model": "text-davinci-003",
}
}'
'''
from djl_python.open_ai import custom_parse_input


class Test(TestCase):
    def test_custom_parse_input(self):
        prompt = {"prompt": "Bought book on human rights in usa with my card",
                          "top_p": 1,
                          "temperature": 0.7}
        inputs = test_model.create_json_request(prompt)

        a, b, c, d, e = custom_parse_input(self, inputs)
        print(a, b, c, d, e)
