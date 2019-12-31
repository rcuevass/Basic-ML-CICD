import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from fashion_mnist_classifier.load_model import load_model
from fashion_mnist_classifier.build_model import build_model
#from fashion_mnist_classifier.load_model import train_model


class MnistFashionClassifier:
    def __init__(self):
        self.model = build_model()

    def load_weights(self, file):
        self.model = load_model(file)

    def fit_from_directory(self, directory, model_path=None):
        self.model = train_model(
            self.model,
            train_dir=directory,
            val_dir=directory,
            model_path=model_path
        )

    def classify_one(self, file):
        x = img_to_array(load_img(file)) / 255

        pred = self.model.predict(np.expand_dims(x, axis=0))

        genre = 'metal' if pred[0][0] < 0.5 else 'not_metal'

        return genre, pred[0][0]
