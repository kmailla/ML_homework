import json
import requests
import run_server
import unittest

MODEL_PATH = '../saved_models/classifier_model.h5'
KERAS_REST_API_URL = "http://localhost:5002/predict"


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run_server.run_model(MODEL_PATH)

    @staticmethod
    def load_from_file(json_file_path):
        with open(json_file_path) as f:
            data = json.load(f)
        return data

    def test_predict_with_correct_input(self):
        correct_request = self.load_from_file('test_data/correct_request.json')
        response = requests.post(KERAS_REST_API_URL, json=correct_request).json()
        expected_success = 'true'

        self.assertEqual(response['success'], expected_success)

    def test_predict_with_wrong_key(self):
        wrong_key_request = self.load_from_file('test_data/wrong_key_request.json')
        response = requests.post(KERAS_REST_API_URL, json=wrong_key_request).json()
        expected_error = "The request should contain 'data_point' as a key"

        self.assertEqual(response['error'], expected_error)

    def test_predict_with_wrong_shape(self):
        wrong_shape_request = self.load_from_file('test_data/wrong_shape_request.json')
        response = requests.post(KERAS_REST_API_URL, json=wrong_shape_request).json()
        expected_error = "Accepted input shape: (1, 13)"

        self.assertEqual(response['error'], expected_error)

    def test_predict_with_wrong_array_type(self):
        wrong_array_type_request = self.load_from_file('test_data/wrong_array_type_request.json')
        response = requests.post(KERAS_REST_API_URL, json=wrong_array_type_request).json()
        expected_error = "Please make sure that the data_point array only consists of (real) numbers"

        self.assertEqual(expected_error, response['error'])


if __name__ == '__main__':
    unittest.main()
