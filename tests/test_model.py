import model as m
import numpy as np
from tensorflow import keras
import unittest


class ModelTestCase(unittest.TestCase):
    def test_get_model(self):
        model = m.get_model()
        self.assertTrue(isinstance(model, keras.Sequential))

    def test_train_data_with_malformed_input(self):
        self.assertRaises(ValueError, lambda: m.load_train_data('test_data/malformed_training_data.csv'))

    def test_train_data(self):
        x, y = m.load_train_data('test_data/correct_training_data.csv')
        expected_x = np.zeros((1, 1, 13))
        expected_y = np.asarray([[0, 0, 0, 1]])
        self.assertTrue(np.array_equal(x, expected_x, equal_nan=True))
        self.assertTrue(np.array_equal(y, expected_y, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
