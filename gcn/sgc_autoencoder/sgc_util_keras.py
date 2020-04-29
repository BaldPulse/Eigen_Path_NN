from keras import backend as K
from keras.layers import Layer

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
                                      name='kernel',
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
