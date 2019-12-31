#import os
from flask import Flask
#, request, render_template, jsonify
#from fashion_mnist_classifier.load_model import load_model


#fashion_mnist_classifier = load_model(from_s3='S3_URL' in os.environ)

app = Flask(__name__)

# important!!! it is a decorator!
@app.route('/', methods=['GET'])
def index():
	return '<h1>hello world...again<h1>'


@app.route('/predict', methods=['GET'])
def predict():

	return '<h1>hello again!<h1>'


if __name__ == '__main__':
	app.run()