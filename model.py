import json
import numpy as np
from tensorflow import keras


def get_encoder_model():
    """
    This function returns a predefined model for the task.

    :returns: a Keras model
    """
    encoder_model = keras.Sequential([keras.layers.LSTM(units=128, activation='tanh', return_sequences=True,
                                                        input_shape=(None, 13)),
                                      keras.layers.LSTM(units=64, activation='tanh'),
                                      keras.layers.Dense(10)])
    encoder_model.compile(optimizer="adam", loss="mean_squared_error")
    return encoder_model


def load_weights(file_path, model):
    """
    Load weights from the file to a chosen model

    :param file_path: the path to the file containing the weights
    :param model: a Keras model
    :raises keyError: raised if the dictionary read from the file misses the expected keys
    """
    with open(file_path) as json_file:
        weights = json.load(json_file)

    # convert the key values into numpy arrays
    for key, value in weights.items():
        weights[key] = np.asarray(value)

    try:
        # the LSTM bias vector is the sum of the input and hidden bias vectors
        lstm_0_bias = np.add(weights['LSTM_0_inner_hidden_bias'], weights['LSTM_0_hidden_hidden_bias'])
        lstm_1_bias = np.add(weights['LSTM_1_inner_hidden_bias'], weights['LSTM_1_hidden_hidden_bias'])

        weights_list = [
            [weights['LSTM_0_inner_hidden_kernel'],
             weights['LSTM_0_hidden_hidden_kernel'],
             lstm_0_bias],
            [weights['LSTM_1_inner_hidden_kernel'],
             weights['LSTM_1_hidden_hidden_kernel'],
             lstm_1_bias],
            [weights['Dense_2_kernel'],
             weights['Dense_2_bias']]
        ]

        for layer, weights in zip(model.layers, weights_list):
            layer.set_weights(weights)
    except KeyError as e:
        raise KeyError('Missing key from the loaded weights dict: {}'.format(e.args[0]))


