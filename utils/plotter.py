# for plotting
import matplotlib.pyplot as plt
# and arrays
import numpy as np
# for types
from typing import List
#
import tensorflow as tf
import utils


# get logging object
log_aux = utils.get_log_object()


'''
Module used for plotting purposes
'''


def plot_image(index_: int,
               list_test_labels_: List[str],
               predictions_array: np.ndarray,
               true_label: np.ndarray,
               img: np.ndarray) -> None:

    """
    Auxiliary function used  to plot an image from an array of images, based on a given
    index
    :param index_: integer that indicates number of image, from array of images, to be plotted
    :param list_test_labels_: list of strings that indicates labels associated to each image in a given array
    :param predictions_array: numpy array containing predictions made on a set of images
    :param true_label: numpy array containing actual values, as opposed to predictions, made on a set of images
    :param img: numpy array containing a set of images
    :return: None
    """

    # update predictions array, true_labels and images
    # the last two are obtained by sub-setting true_label and img based on index_
    predictions_array, true_label, img = predictions_array, true_label[index_], img[index_]

    # do not show grid
    plt.grid(False)
    # add ticks to X and Y axes
    plt.xticks([])
    plt.yticks([])

    # show image obtained from index_
    plt.imshow(img, cmap=plt.cm.binary)
    # get predicted label as argument of predictions_array
    predicted_label = np.argmax(predictions_array)
    # when displaying in plot, show prediction in blue when prediction is accurate
    # and show it in red when it corresponds to wronged classification
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # add label to plot generated
    plt.xlabel("{} {:2.0f}% ({})".format(list_test_labels_[predicted_label],
                                         100*np.max(predictions_array),
                                         list_test_labels_[true_label]),
               color=color)


def plot_value_array(index_: int,
                     predictions_array: np.ndarray,
                     true_label: np.ndarray) -> None:

    """
    Auxiliary function to display bar plot showing distribution of probabilities
    on a given image in a set of images
    :param index_: integer that indicates number of image, from array of images, to be plotted
    :param predictions_array: numpy array containing predictions made on a set of images
    :param true_label: numpy array containing actual values, as opposed to predictions, made on a set of images
    :return: None
    """

    # Update true_label from the sub-setting of true label based on index_
    predictions_array, true_label = predictions_array, true_label[index_]
    # Switch off grid on plot
    plt.grid(False)
    # Add ticks to X and Y axes
    plt.xticks(range(10))
    plt.yticks([])

    # Get bar plot based on predictions_array; it provides as probability distribution
    bar_plot_to_show = plt.bar(range(10), predictions_array, color="#777777")
    # set limit in Y axis between 0 and 1.02; a bit wider than ranges of probability distributions
    plt.ylim([0, 1.02])
    # get argument of predicted arrays
    predicted_label = np.argmax(predictions_array)
    # show in red predicted values...
    bar_plot_to_show[predicted_label].set_color('red')
    # ... and show in blue true values
    bar_plot_to_show[true_label].set_color('blue')


def plot_graphs(history: tf.keras.callbacks.History,
                plot_title: str,
                string_metric: str,
                location_plot: str) -> None:

    """
    Function that generates plot of loss and accuracy vs epoch
    :param history: TF2 History object that comes from model fitting
    :param plot_title: string to be used as plot title
    :param string_metric: string that captures the metric to be plotted vs epochs
    :param location_plot: string that has location where plot will be saved
    :return: None
    """

    # add title to plot
    plt.title(plot_title)
    # if there is a value assigned to string_metric, add plot of such metric
    # vs. epoch, for both training and validation
    if string_metric != '':
        plt.plot(history.history[string_metric])
        plt.plot(history.history['val_'+string_metric])
    # Add plot of loss vs epoch for both training and validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_' + 'loss'], '')
    # add label to X axis
    plt.xlabel("Epochs")
    # if there is a value assigned to string_metric, add legend plot of such metric
    # vs. epoch, for both training and validation and similar for loss in both
    # datasets. Only add plot of loss otherwise
    if string_metric != '':
        plt.legend([string_metric, 'val_'+string_metric, 'loss', 'val_loss'])
    else:
        plt.legend(['loss', 'val_loss'])

    # add grid to plot
    plt.grid(True, which='major', linestyle='-')
    # set vertical range to [0, 1.02]
    plt.gca().set_ylim(0, 1.02)
    # save plot to file and display message for user
    plt.savefig(location_plot+'performance_'+plot_title+'.png')
    print('image saved to ', location_plot+'performance_'+plot_title+'.png')
    # clears plot to avoid overlap with future plots coming from future calls
    plt.clf()
    #plt.show()


def plot_images_fashion_mnist(images: np.ndarray,
                              images_labels: np.ndarray,
                              file_location: str,
                              number_of_images_per_row: int = 5) -> None:
    """
    Function that plots MNIST images in an array of
    number_of_images_per_row X number_of_images_per_row
    :param images: numpy array containing images to be plotted
    :param images_labels: numpy array with corresponding labels of passed images
    :param file_location: string with folder location where images will be saved
    :param number_of_images_per_row: integer setting the number of images per row to show
                                    default value = 5
    :return: None
    """

    # list of classes for images
    list_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                        'Ankle boot']

    # get total number of images to be displayed from number of
    # images per row
    number_of_images: int = number_of_images_per_row*number_of_images_per_row

    # set size of figure
    plt.figure(figsize=(9, 9))
    # loop over number of images
    # the square array of images is populated from left to right and from
    # top to bottom
    for i in range(number_of_images):
        # Create subplot according to the number of images per row
        plt.subplot(number_of_images_per_row,
                    number_of_images_per_row,
                    i+1)
        # add ticks to X and Y axes
        plt.xticks([])
        plt.yticks([])
        # Do not include grid on image
        plt.grid(False)
        # Add image in turn (i-th image)
        plt.imshow(images[i], cmap=plt.cm.binary)
        # add class name of the image in turn
        plt.xlabel(list_class_names[images_labels[i]])
    # save plot to file
    plt.savefig(file_location + 'sample_images.png')
    # clears plot to avoid overlap with future plots coming from future calls
    plt.clf()
    #plt.show()


def evaluate_model_plot_predictions(model_: tf.keras.models.Sequential,
                                    model_name: str,
                                    train_set: List[np.ndarray],
                                    test_set: List[np.ndarray],
                                    num_epochs: int,
                                    location_plot: str) -> None:

    """
    Function that evaluates a deep and wide regression model; generates plot to show performance
    graphically and adds a bar plot to show probability distribution for classes
    :param model_: TF2 sequential model
    :param model_name: string that labels name of given model
    :param train_set: numpy array that captures train set
    :param test_set: numpy array that captures test set
    :param num_epochs: integer that captures number of epochs
    :param location_plot: string that indicates where plot will be saved to
    :return: None
    """

    # names of classes for dataset
    list_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                        'Ankle boot']

    # compile and fit model
    model_.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    # Create a tf.keras.callbacks.ModelCheckpoint callback that saves
    # weights only during training
    checkpoint_path = '../models/' + str(model_name) + '/' + model_name + '.ckpt'

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model_.fit(train_set[0], train_set[1],
                         validation_data=(test_set[0], test_set[1]),
                         epochs=num_epochs,
                         callbacks=[cp_callback])

    path_to_save_model = '../models/' + str(model_name) + '/' + model_name + '.h5'
    log_aux.info('saving model to=%s', path_to_save_model)
    model_.save(path_to_save_model)

    # evaluate on test set
    model_.evaluate(test_set[0], test_set[1], verbose=2)

    # generate plots associated with models performance
    plot_graphs(history, plot_title=model_name, string_metric='accuracy',
                location_plot=location_plot)

    # set test labels and test images
    test_labels = test_set[1]
    test_images = test_set[0]
    # make predictions
    predictions = model_.predict(test_images)

    # set file_name based on model_name
    file_name = location_plot + model_name + '_predictions.png'

    # set number of rows and columns for the plot to be generated
    num_rows: int = 5
    num_cols:int = 3
    # get total number of images
    num_images: int = num_rows * num_cols
    # set size of figure
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    # loop over total number of images...
    for i in range(num_images):
        # on subplot add image in turn
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(index_=i, list_test_labels_=list_class_names,
                   predictions_array=predictions[i],
                   true_label=test_labels,
                   img=test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(index_=i, predictions_array=predictions[i],
                         true_label=test_labels)

    # set layout and save figure to file...
    plt.tight_layout()
    plt.savefig(file_name)
    print('image saved to ', file_name)
    log_aux.info('image saved to=%s', file_name)
    # clears plot to avoid overlap with future plots coming from future calls
    plt.clf()
    #plt.show()


