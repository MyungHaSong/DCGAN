import tensorflow as tf

def dc_conv(input,filters,name, kernel_size = (5,5),  strides =(2,2), padding = "SAME", is_train = True):
    net = tf.layers.conv2d(input, filters = filters, kernel_size = kernel_size, strides = strides, padding =padding ,trainable= is_train  )
    return net

def dc_deconv(input,filters,name, kernel_size=(5,5),strides =(2,2), padding ="SAME",is_train = True ):
    net = tf.layers.conv2d_transpose(input, filters = filters, kernel_size=kernel_size, strides = strides, padding = padding , trainable= is_train)
    return net

def dc_dense(input, units ,name, is_train = True):
    net = tf.layers.dense(input, units = units, trainable = is_train)
    return net

def flatten(input):
    net = tf.layers.flatten(input)
    return net

def batch_norm(input):
    net = tf.layers.batch_normalization(input,training = True)
    return net
