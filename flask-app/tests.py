import imageio
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def make_prediction(path_to_image: str) -> str:

    # list of classes for prediction
    list_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                        'Ankle boot']

    # load model and image
    model_ = load_model('../models/model_03/model_03.h5')
    image_ = imageio.imread(path_to_image)
    #image_ = img_to_array(load_img(path_to_image))

    # reshape image by expanding dimensions
    image_ = np.expand_dims(image_, axis=0)

    # make prediction
    prediction_value = model_.predict(image_)[0]

    # find position with maximum probability
    max_probability_position = int(np.argmax(prediction_value))

    # get label
    predicted_label = list_class_names[max_probability_position]

    return predicted_label


predicted_string_label = make_prediction('../data/test/image_30.jpg')
print(predicted_string_label)

