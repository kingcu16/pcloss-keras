import os
from keras.layers import *
from keras.models import Model
import keras.backend as K
from PCLoss import PCLoss


def create_model(input_shape=(128,128,3), classes=100):
    inputs = Input(shape=input_shape, name='input')
    label  = Input(shape=(classes,), name='label')
    x = Conv2D(64, (3,3), padding='same', name='conv_1')(inputs)
    x = Conv2D(128, (3,3), padding='same', name='conv_2')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Activation('relu', name='relu_1')(x)

    x = Conv2D(128, (3,3), padding='same', name='conv_3')(x)
    x = Conv2D(128, (3,3), padding='same', name='conv_4')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Activation('relu', name='relu_2')(x)

    x = MaxPool2D()(x)

    x = conv_block(128, (7,7), block_name='block_1')(x)
    pc = GlobalAvgPool2D(name='gap1')(x)
    pc1 = PCLoss(classes=classes, use_bias=False, name='pc_loss1')(pc, label=label)
    x = MaxPool2D()(x)
    x = conv_block(256, (9,9), block_name='block_2')(x)
    pc = GlobalAvgPool2D(name='gap2')(x)
    pc2 = PCLoss(classes=classes, use_bias=False,  name='pc_loss2')(pc, label=label)
    x = MaxPool2D()(x)
    x = conv_block(512, (13, 13), block_name='block_3')(x)
    x = GlobalAvgPool2D(name='gap3')(x)
    pc3 = PCLoss(classes=classes, use_bias=False,  name='pc_loss3')(x,label=label)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(classes)(x)
    x = Activation('softmax', name='classes')(x)
    ce_loss = Lambda(lambda x:K.categorical_crossentropy(label,x), output_shape=(1,))(x)
    pc_loss = Add(name='pc_loss')([pc1, pc2, pc3])

    return Model([inputs, label], [pc_loss, ce_loss])


def conv_block(filters, kernel_size,padding='same', strides=(1,1),block_name='block_1',**kwargs):
    def res_block(x1):
        x = Conv2D(filters, kernel_size, strides=strides,padding=padding, name=block_name+'_conv_1', **kwargs)(x1)
        x = Conv2D(filters, kernel_size, strides=strides,padding=padding, name=block_name+'_conv_2', **kwargs)(x)
        x = BatchNormalization(name=block_name+'_bn_1',)(x)
        x1 = Conv2D(filters, (1,1), name=block_name+'_11')(x1)
        x = Add()([x1, x])
        x = Activation('relu', name=block_name+'_relu_1')(x)
        return x
    return res_block

if __name__ == "__main__":
    model = create_model()
    model.summary()