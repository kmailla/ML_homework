# ML take home test
This repository contains source code for the ML take home test I was assigned to.

The tasks were the following:
* create and load weights into an encoder based on a given weights file
* extend the model with a decoder part and train it on a given labelled data file
* create an API that accepts relevant input data and return
    * the encoding vector
    * the predicted label
    * the model score
    
## How to try it
The model can be reached through this endpoint: https://ml-homework.herokuapp.com/predict

For documentation, please visit https://ml-homework.herokuapp.com/docs

**Important note:** I'm using a free app on Heroku, so there might be a delay or an application error returned when 
sending the first request as the app goes sleeping after 30 minutes of inactivity. 
Just try again after the first request.

Example request:
```
curl -X 'POST' \
  'https://ml-homework.herokuapp.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data_point": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
}'
```
The above request gets the following response:
```
{
  "success": true,
  "predictedLabel": 0,
  "modelScore": 0.8781212568283081,
  "encodingVector": [
    [
      0.9860631227493286,
      1.6803410053253174,
      0.5341314077377319,
      -0.7231469750404358,
      -0.7793399691581726,
      0.46401655673980713,
      -0.25720030069351196,
      -0.07891435921192169,
      0.3423164188861847,
      0.6402440667152405
    ]
  ]
}
```

## Running locally
The prerequisites for running this project locally is having Python3.8+ installed along with python3-venv:
```
apt install python3-venv
```
After that, navigate to the root of the project and start the `setup.sh` script:
```
bash setup.sh
```
This will create a virtual environment, download the packages needed and also spin up the server on http://localhost:5002

Now we can send an example request:
```
curl -X 'POST' \
  'http://localhost:5002/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"data_point": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}'`
```

## Run tests
In order to run the tests, run the following commands in the root of the project:
```
python -m unittest tests.test_model
python -m unittest tests.test_server
```