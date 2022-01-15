import json
from keras.callbacks import EarlyStopping
from keras.models import save_model
from keras.layers import Dense, LSTM
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

PRETRAINED_WEIGHTS_FILE = 'data/weights.json'
TRAINING_DATA = 'data/labelled.csv'


def get_model():
    """
    This function returns a predefined model for the task.
    The model can be used as a classifier and consist of an encoder part (which can be loaded with the pretrained
    weights) and a decoder.

    :returns: a Keras sequential model
    """
    model = keras.Sequential([LSTM(units=128, activation='tanh', return_sequences=True, input_shape=(None, 13)),
                              LSTM(units=64, activation='tanh'),
                              Dense(10),
                              # decoder part
                              Dense(4, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_weights(file_path, model):
    """
    Loads weights from a file to a chosen model. It excepts key names consistent to `weights.json`.

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


def load_train_data(file_path):
    """
    Loads the training data, excepts a comma separated file with a header and an empty last line.

    :param file_path: the path to the file containing the labelled data
    :raises valueError: raised if the loaded data does not have the appropriate shape
    """
    x = []
    y = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    # ignore the first and last lines
    for line in lines[1:-1]:
        data_points_str = line.split(',')
        if len(data_points_str) != 14:
            raise ValueError('Invalid input data shape, expecting 14 data point per line')
        # convert string values to float
        data_points = [float(d) for d in data_points_str]
        x.append([data_points[:-1]])
        y.append([data_points[-1]])

    # convert the nested lists into numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    # one hot encode class labels
    y = to_categorical(y, num_classes=4)

    return x, y


def train_model(model, x, y):
    """
    Trains a model with the accompanying data

    :param model: a Keras model
    :param x: features for training and validation
    :param y: labels for training and validation
    """
    # extract a validation set from the data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, shuffle=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    model.fit(x_train, y_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_val, y_val),
              callbacks=[early_stop])


def create_new_model(model_name):
    """
    Creates and saves a new model by using the files given in the take home test

    :param model_name: a custom name to use when saving the model
    """
    encoder = get_model()
    load_weights(PRETRAINED_WEIGHTS_FILE, encoder)
    x, y = load_train_data(TRAINING_DATA)
    train_model(encoder, x, y)
    save_model(encoder, './saved_models/' + model_name + '.h5')
