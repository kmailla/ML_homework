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
