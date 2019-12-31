from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import numpy as np
import utils
#import scipy.misc
import imageio


# get logging object
log = utils.get_log_object()


def train_from_dictionary(dictionary_of_models,
                          num_epochs, plot_location):

    dictionary_predictions_ = dict()

    for model_ in dictionary_of_models.keys():
        log.info('Evaluating model=%s', model_)
        model_name = model_
        model_object = dictionary_models[model_name]

        utils.evaluate_model_plot_predictions(model_=model_object,
                                              model_name=model_name,
                                              train_set=[train_images, train_labels],
                                              test_set=[test_images, test_labels],
                                              num_epochs=num_epochs, location_plot=plot_location)

        log.info('Evaluation of model has been completed=%s', model_)

    return dictionary_predictions_


if __name__ == '__main__':

    # set number of epochs
    number_of_epochs: int = 16
    # get data and split
    fashion_mnist = keras.datasets.fashion_mnist
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # save images to test folder
    number_of_test_images: int = test_images.shape[0]
    for i in range(100):

        image_array = test_images[i]
        #img_uint8 = image_array.astype(np.uint8)
        file_image_name = '../data/test/image_' + str(i) + '.jpg'
        imageio.imwrite(file_image_name, image_array)

    print('done!')
    quit()

    #
    utils.plot_images_fashion_mnist(images=train_images,
                                    images_labels=train_labels,
                                    file_location='../plots/fashion_mnist/')

    dictionary_models = dict()
    dictionary_models['model_01'] = utils.model_01()
    dictionary_models['model_02'] = utils.model_02()
    dictionary_models['model_03'] = utils.model_03()

    dictionary_models_predictions =\
        train_from_dictionary(dictionary_of_models=dictionary_models,
                              num_epochs=number_of_epochs,
                              plot_location='../plots/fashion_mnist/')