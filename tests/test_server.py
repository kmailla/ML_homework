try:
    import run_server
except ImportError:
    from .. import run_server
import json
from fastapi.testclient import TestClient
import os
import unittest


client = TestClient(run_server.app)

FOLDER = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(FOLDER, '..')) + '/saved_models/classifier_model.h5'
PREDICTION_API_URL = "http://localhost:5002/predict"


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
        correct_request = self.load_from_file(FOLDER + '/test_data/correct_request.json')
        print('Loaded request.')
        response = client.post(PREDICTION_API_URL, json=correct_request).json()
        expected_success = True

        self.assertEqual(expected_success, response['success'])

    def test_predict_with_wrong_key(self):
        wrong_key_request = self.load_from_file(FOLDER + '/test_data/wrong_key_request.json')
        response = client.post(PREDICTION_API_URL, json=wrong_key_request).json()
        expected_error = "value_error.missing"

        self.assertEqual(expected_error, response['detail'][0]['type'])

    def test_predict_with_wrong_shape(self):
        wrong_shape_request = self.load_from_file(FOLDER + '/test_data/wrong_shape_request.json')
        response = client.post(PREDICTION_API_URL, json=wrong_shape_request).json()
        expected_error = "Accepted input shape: (1, 13)"

        self.assertEqual(expected_error, response['detail'])

    def test_predict_with_wrong_array_type(self):
        wrong_array_type_request = self.load_from_file(FOLDER + '/test_data/wrong_array_type_request.json')
        response = client.post(PREDICTION_API_URL, json=wrong_array_type_request).json()

        # the API first checks if the type is int then if it's float
        first_expected_error = "type_error.integer"
        second_expected_error = "type_error.float"

        self.assertEqual(first_expected_error, response['detail'][0]['type'])
        self.assertEqual(second_expected_error, response['detail'][1]['type'])


if __name__ == '__main__':
    unittest.main()
