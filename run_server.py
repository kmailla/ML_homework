import argparse
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from keras.models import load_model
from model import get_encoding_vector
import numpy as np
from pydantic import BaseModel
from typing import List, Union
import uvicorn


# initialize Flask application
app = FastAPI()
model = None
MODEL_PATH = 'saved_models/classifier_model.h5'


class HTTPError(BaseModel):
    success: bool = False
    detail: str

    class Config:
        schema_extra = {
            "example": {"detail": "Some error occurred"},
        }


class Response(BaseModel):
    success: bool = True
    predictedLabel: int
    modelScore: float
    encodingVector: List[List[float]]


class Request(BaseModel):
    data_point: List[List[Union[int, float]]]

    class Config:
        schema_extra = {
            "example": {"data_point": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]},
        }


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


@app.post('/predict',
          responses={
              200: {"model": Response},
              406: {
                  "model": HTTPError
              },
          },
          )
def predict(request: Request):
    """
    Reads the arriving requests, processes it into a numpy array and sends the predicted label and probability
    """
    # check if the model is running
    if model is None:
        run_model()

    # here we add the third dimension
    data_point = np.asarray([request.data_point])

    # only accept the correct shape for the data
    if data_point.shape != np.zeros((1, 1, 13)).shape:
        # the error message to the user warns about only two dimensions
        raise HTTPException(406, detail="Accepted input shape: (1, 13)")

    # run prediction and return the most probable label along with the score
    preds = model.predict(data_point)

    return {
        'success': True,
        'predictedLabel': int(np.argmax(preds)),
        'modelScore': float(max(preds[0])),
        'encodingVector': get_encoding_vector(model, data_point).tolist()
    }


# load the model and then start the server
# this function is used when starting the server by executing this python file
if __name__ == '__main__':
    print("* Loading Keras model and Flask starting server...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to the model')
    args = parser.parse_args()

    # expecting the model path as a first arg
    custom_model_path = args.model_path
    run_model(custom_model_path)
    uvicorn.run(app, host="localhost", port=5002)
