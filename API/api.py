from keras.applications.mobilenet import MobileNet
import tensorflow as tf
from flask import Flask
from flask import request
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
app = Flask(__name__)

# Load MobileNet
MobileNet = MobileNet()

# The following line is necessary because Flask runs on a different thread than the loaded model
graph = tf.get_default_graph()

# Load the deep convolutional recurrent extra crispy one neuron network
LinearRegression = LinearRegression()
LinearRegression.fit([[1], [2]], [[2], [4]])


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    # Get input value
    X = int(request.form['X'])

    # Place input value in a list of lists
    X = [[X]]

    # Predict
    prediction = LinearRegression.predict(X)

    # Return prediction
    return str(prediction)


@app.route('/predict_mobilenet', methods=['POST'])
def predict_mobilenet():
    # Get file descriptor
    file = request.files['input']

    # Read bytestream from file
    string = file.read()

    # Turn bytestream into 1D array
    data = np.fromstring(string, dtype='uint8')

    # Turn 1D array into 2D image
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Resize image so it fits model input
    image = cv2.resize(image, (224, 224))

    # Reshape image so it fits model input
    image = np.reshape(image, (1, *image.shape))

    # The following line is necessary because Flask runs on a different thread than the loaded model
    with graph.as_default():
        # Predict
        prediction = MobileNet.predict(image)

    # Return prediction
    return str(prediction)


if __name__ == '__main__':
    app.run()
