from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import skimage
import numpy as np

class sgc_decoder(Layer):

    def __init__(self,
                 output_dim,
                 activation=None,
                 use_bias=True,                 
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None
        super(sgc_decoder, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        sm_kernel = K.softmax(self.kernel)
        return K.dot(x, sm_kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    import tensorflow as tf

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.summary.image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img = data.astronaut()
        # Do something to the image
        img = (255 * skimage.util.random_noise(img)).astype('uint8')

        image = make_image(img)
        summary = tf.summary(value=[tf.summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return

import tensorflow as tf



class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        decoder_weight = self.model.get_layer('decoder').get_weights()
        img = tf.squeeze(decoder_weight)
        shape = K.int_shape(img)
        img = tf.reshape(img, [1, shape[0], shape[1], 1])
        file_writer = tf.summary.create_file_writer('./log')
        with file_writer.as_default():
            tf.summary.image("Decoder Paths", img, step=5, description='paths generated from softmaxing decoder weights')

        return
