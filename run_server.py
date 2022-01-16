import argparse
from keras.models import load_model
from model import get_encoding_vector
import numpy as np
import flask

# initialize Flask application
app = flask.Flask(__name__)
model = None
MODEL_PATH = 'saved_models/classifier_model.h5'


def run_model(model_path=MODEL_PATH):
    """
    Loads an existing prediction model

    :param model_path: the path to the saved model, falls back to a fixed path when not given
    """
    global model
    # load weights into new model
    if model_path is None:
        model_path = MODEL_PATH
    model = load_model(model_path)
    print("Model loaded.")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Reads the arriving requests, processes it into a numpy array and sends the predicted label and probability
    """
    # the expected input is a JSON
    json_msg = flask.request.get_json(force=True)

    if 'data_point' not in json_msg:
        return flask.jsonify({
            'success': 'false',
            'error': "The request should contain 'data_point' as a key"
        })

    # here we add the third dimension
    data_point = np.asarray([json_msg['data_point']])
    print(data_point)
    # the prediction only works on numerical arrays
    if np.isreal(data_point):
        return flask.jsonify({
            'success': 'false',
            'error': "Please make sure that the data_point array only consists of (real) numbers"
        })

    # only accept the correct shape for the data
    if data_point.shape != np.zeros((1, 1, 13)).shape:
        # the error message to the user warns about only two dimensions
        return flask.jsonify({
            'success': 'false',
            'error': "Accepted input shape: (1, 13)"
        })

    # run prediction and return the most probable label along with the score
    preds = model.predict(data_point)

    return flask.jsonify({
        'success': 'true',
        'predictedLabel': int(np.argmax(preds)),
        'modelScore': float(max(preds[0])),
        'encodingVector': get_encoding_vector(model, data_point).tolist()
    })


# load the model and then start the server
if __name__ == '__main__':
    print("* Loading Keras model and Flask starting server...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to the model')
    args = parser.parse_args()

    # expecting the model path as a first arg
    custom_model_path = args.model_path
    run_model(custom_model_path)
    app.config['JSON_AS_ASCII'] = False
    app.run(port=5002)
