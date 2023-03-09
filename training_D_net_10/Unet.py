import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, concatenate, BatchNormalization 
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import add

RESIDUAL = False
BASE_FILTER = 16
DEPTH = 3
FILTER_GROW = True
NUM_CLASS = 2

def up_sampling(input_tensor, scale):
    net = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(scale, scale), interpolation='bilinear'))(input_tensor)
    net = tf.keras.layers.Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
    net = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(scale, 1), interpolation='bilinear'))(net)
    net = tf.keras.layers.Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
    return net

def myConv(x_in, nf, strides=1, kernel_size = 3):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    x_out = Conv3D(nf, kernel_size=kernel_size, padding='same', strides=strides)(x_in)
    x_out = BatchNormalization()(x_out)
    x_out = tf.keras.layers.SpatialDropout3D(0.25)(x_out)
    return x_out


def unet3dBlock(l, n_feat, depth):
    if RESIDUAL:
        l_in = l
    for i in range(2):
        l = myConv(l, n_feat)
    return add([l_in, l]) if RESIDUAL else l


def unetUpsample(l, num_filters):
    l = up_sampling(l, scale=2)
    l = myConv(l, num_filters, kernel_size = 2)
    return l


def unet3d(main_input):
    inputs = main_input
    depth = DEPTH
    filters = []
    down_list = []
    x = myConv(inputs, BASE_FILTER)
    layer = add([x, inputs])
    
    for d in range(depth - 1):
        if FILTER_GROW:
            num_filters = BASE_FILTER * (2**d)
        else:
            num_filters = BASE_FILTER
        filters.append(num_filters)
        layer = unet3dBlock(layer, num_filters, d + 1)        
        if d != depth - 1:
            down_list.append(layer)
            layer = myConv(layer, num_filters*2, strides=2)
        
    for d in range(depth-2, -1, -1):
        layer = unetUpsample(layer, filters[d])
        layer = concatenate([layer, down_list[d]])
        layer = unet3dBlock(layer, filters[d] * 2, d + 1)
        '''
        if configTrain.DEEP_SUPERVISION:
            if 0< d < 3:
                pred = myConv(layer, configTrain.NUM_CLASS)
                if deep_supervision is None:
                    deep_supervision = pred
                else:
                    deep_supervision = add([pred, deep_supervision])
                deep_supervision = UpSampling3D()(deep_supervision)
        '''
    
    '''
    if configTrain.DEEP_SUPERVISION:
        layer = add([layer, deep_supervision])
    '''
    layer = myConv(layer, 1, kernel_size = 1)
    x = Activation('sigmoid', name='sigmoid')(layer)
        
    model = Model(inputs=[inputs], outputs=[x])
    return model