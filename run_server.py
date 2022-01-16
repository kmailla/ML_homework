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
    # initialize the response message
    data = {'success': False}

    if flask.request.method == "POST":
        # the expected input is a JSON
        json_msg = flask.request.get_json(force=True)

        try:
            # here we add the third dimension
            data_point = np.asarray([json_msg['data_point']])
        except KeyError:
            data['error_msg'] = "The request should contain 'data_point' as a key"
            return flask.jsonify(data)

        # only accept the correct shape for the data
        if data_point.shape != np.zeros((1, 1, 13)).shape:
            # the error message to the user warns about only two dimensions
            data['error_msg'] = "Accepted input shape: (1, 13)"

        else:
            # run prediction and return the most probable label along with the score
            preds = model.predict(data_point)
            data['predicted_label'] = int(np.argmax(preds))
            data['model_score'] = float(max(preds[0]))

            # add the encoding vector to the message
            data['encoding_vector'] = get_encoding_vector(model, data_point).tolist()

            # indicate that the request was a success
            data['success'] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# load the model and hen start the server
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
