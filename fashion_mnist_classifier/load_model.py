import tempfile
import wget
import os
from fashion_mnist_classifier.fashion_mnist_classifier import MnistFashionClassifier


def load_model(from_s3=True):
    classifer_ = MnistFashionClassifier()

    if from_s3:
        s3_url = os.getenv('S3_URL')
        with tempfile.TemporaryDirectory() as tmp_dir:
            print('Fetching weights from S3', s3_url)
            wget.download(s3_url, '{}/weights.h5'.format(tmp_dir))

            print('Loading weights...')
            classifer_.load_weights('{}/weights.h5'.format(tmp_dir))
    else:
        print('Loading model from file system...')
        classifer_.load_weights('../models/model_03/model_03.h5')

    return classifer_
