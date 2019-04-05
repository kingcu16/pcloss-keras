from keras.layers import *
import keras.backend as K

class PCLoss(Layer):
    '''
    PC Loss Implementation by Keras
    example:
        ...
        pcloss = PCLoss(classes=1000)(x, label=label)
        ...
        model = Model(inputs, pcloss)

        model.compile(...,
                loss = lambda ytrue, ypred: ypred,
                ...)

    '''
    def __init__(self, classes,
                use_bias=True,
                use_celoss = True,
                use_pcloss = True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(PCLoss, self).__init__(**kwargs)
        self.classes = classes
        self.use_bias = use_bias
        self.use_celoss = use_celoss
        self.use_pcloss = use_pcloss
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.classes),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.classes,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self, inputs, label):
        ce_loss = 0
        pc_loss = 0
        if self.use_celoss:
            output = K.dot(inputs, self.kernel)
            if self.use_bias:
                output = K.bias_add(output, self.bias, data_format='channels_last')
            output = K.softmax(output)
            ce_loss = K.categorical_crossentropy(label, output)
        if self.use_pcloss:    
            input_shape = K.shape(inputs)

            weights_t = K.dot(label, K.transpose(self.kernel))
            pcl1 = K.sum(K.sqrt(K.square(inputs -weights_t )), axis=-1)
            
            expand_weights = K.expand_dims(K.transpose(self.kernel), 0)
            expand_inputs = K.expand_dims(inputs, 1)
            tile_expand_weights = K.tile(expand_weights, [input_shape[0], 1,1])
            tile_expand_inputs = K.tile(expand_inputs, [1, self.classes, 1])
            diff_input_weights = tile_expand_inputs - tile_expand_weights
            expand_weights_t = K.expand_dims(weights_t, 1)
            tile_expand_weights_t = K.tile(expand_weights_t, [1, self.classes, 1])
            diff_weights_weights = tile_expand_weights_t - tile_expand_weights
            pcl2 = K.sum(K.sqrt(K.square(diff_input_weights)), axis=-1) + K.sum(K.sqrt(K.square(diff_weights_weights)), axis=-1)
            pcl2 = pcl2 * (1-label)
            pcl2 = K.sum(pcl2, axis=-1)  / (self.classes-1) 
            pc_loss = pcl1 - pcl2

        return pc_loss + ce_loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
