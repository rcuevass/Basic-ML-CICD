import imageio
import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
# TODO - last TF import does not seem to work in FLASK development environment -- figure it out!!!


# list of classes for prediction
list_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('Loading model...')
print('Current dir', os.getcwd())
model_ = load_model('../models/model_03/model_03.h5')
df_test_labels = pd.read_csv('../data/labels_top_100_test_images.csv')
print('Model and labels loaded...')


app = Flask(__name__)

# important!!! it is a decorator!
@app.route('/', methods=['GET'])
def index():
	return render_template('index.html', name='Simple TF2 classifier')


@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		if 'mnist_image' not in request.files:
			return 'No file found', 400

		file_ = request.files['mnist_image']

		try:
			# read file and reshape by expanding
			image_ = imageio.imread(file_)
			image_ = np.expand_dims(image_, axis=0)

		except:
			return 'Invalid input', 400

		string_file = str(file_)
		string_file = string_file.replace("<FileStorage: 'image_", "").replace(".jpg' ('image/jpeg')>", "")
		#string_file = string_file.replace(".jpg' ('image/jpeg')>", "")
		dict_labels = df_test_labels.to_dict()['Label']

		# get actual label
		actual_label = dict_labels[int(string_file)]

		# make prediction
		prediction_value = model_.predict(image_)[0]

		# find position with maximum probability
		max_probability_position = int(np.argmax(prediction_value))

		# get label
		predicted_label = list_class_names[max_probability_position]

		print('file name...', file_)
		print('actual clothing...', actual_label)
		print('predicted clothing...', predicted_label)

		return jsonify({'predicted image': predicted_label, 'actual image': actual_label})


if __name__ == '__main__':
	app.run()