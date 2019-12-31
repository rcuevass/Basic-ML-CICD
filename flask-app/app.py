from flask import Flask, request, render_template
#from tensorflow.keras.models import load_model

#from tensorflow.keras.preprocessing.image import load_img, img_to_array


'''

#from fashion_mnist_classifier.load_model import load_model

#import tensorflow as tf


#import os
#, request, render_template, jsonify
#from fashion_mnist_classifier.load_model import load_model
#fashion_mnist_classifier = load_model(from_s3='S3_URL' in os.environ)
#model = load_model('../models/model_03/model_03.h5')

'''

#model = load_model('../models/model_03/model_03.h5')

app = Flask(__name__)

# important!!! it is a decorator!
@app.route('/', methods=['GET'])
def index():
	#return '<h1>hello world ... again ... kaka!!! <h1>'
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	#model = load_model('../models/model_03/model_03.h5')
	if 'mnist_image' not in request.files:
		return 'Bad request', 400

	print(request.files['mnist_image'])
	return 'hello there again', 200


if __name__ == '__main__':
	app.run()


# /Users/rcuevas/PycharmProjects/Basic-ML-CICD/data/test/image_10.jpg
# http://localhost:5000/predict