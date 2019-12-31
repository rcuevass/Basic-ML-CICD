import tensorflow as tf

''' Module where DL/TF models are defined'''


def build_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # hidden layer with relu activation function and 128 hidden units
        tf.keras.layers.Dense(128, activation='relu'),
        # hidden layer with relu activation function and 100 hidden units
        # dropout rate = 0.2
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # hidden layer with relu activation function and 50 hidden units
        # dropout rate = 0.2
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # hidden layer with relu activation function and 100 hidden units
        # dropout rate = 0.2
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # hidden layer with relu activation function and 128 hidden units
        tf.keras.layers.Dense(128, activation='relu'),
        # softmax activation function to be consistent with
        # multiclass classification - 10 classes; one per digit
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
