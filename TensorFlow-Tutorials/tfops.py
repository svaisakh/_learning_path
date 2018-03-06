import numpy as np
import tensorflow as tf
from utils import dict_to_list

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
def activate(prev_layer, activation='None', name='activation'):
    with tf.name_scope(name):
        if activation=='relu':
            layer = tf.nn.relu(prev_layer)
        elif activation=='sigmoid':
            layer = tf.nn.sigmoid(prev_layer)
        elif activation=='tanh':
            layer = tf.nn.tanh(prev_layer)
        elif activation=='softplus':
            layer = tf.nn.softplus(prev_layer)
        elif activation=='lrelu':
            layer = lrelu(prev_layer)
        elif activation=='softmax':
            layer = tf.nn.softmax(prev_layer)
        else:
            layer = prev_layer
    
        return layer

def fc(prev_layer, num_neurons, activation='relu', name="fc", std=None):
    with tf.variable_scope(name):
        with tf.name_scope('out_shape_calc'):
            prev_neurons = int(prev_layer.shape[1])
        
        if not std:
            std = np.sqrt(2 / prev_neurons)
            
        weights = tf.get_variable('weights', (prev_neurons, num_neurons), initializer=tf.truncated_normal_initializer(stddev=std))
        biases = tf.get_variable('biases', [num_neurons], initializer=tf.zeros_initializer())

        outputs = tf.matmul(prev_layer, weights) + biases

        layer = activate(outputs, activation)

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('outputs', outputs)
        tf.summary.histogram('activations', layer)

        return layer
    
def conv(prev_layer, num_filters, activation='relu', pool=False, reduce_size=True, name="conv", std=None):
    with tf.variable_scope(name):
        with tf.name_scope('out_shape_calc'):
            prev_channels = int(prev_layer.get_shape()[3])
            
        if not std:
            std = np.sqrt(2 / prev_channels)
        
        weights = tf.get_variable('weights', (5, 5, prev_channels, num_filters), initializer=tf.truncated_normal_initializer(stddev=std))
        biases = tf.get_variable('biases', [num_filters], initializer=tf.zeros_initializer())
        
        if pool:
            outputs = tf.nn.conv2d(prev_layer, weights, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + biases
            layer = activate(outputs, activation)
            layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        elif reduce_size:
            outputs = tf.nn.conv2d(prev_layer, weights, strides=[1, 2, 2, 1], padding='SAME', name='convolution') + biases
            layer = activate(outputs, activation)
        else:
            outputs = tf.nn.conv2d(prev_layer, weights, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + biases
            layer = activate(outputs, activation)

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('outputs', outputs)
        tf.summary.histogram('activations', layer)

        return layer
    
def deconv(prev_layer, num_filters, dim_upsampling_ratio=2, activation='relu', out_shape=None, name="deconv", std=None, filter_size=5, stride=2, padding='SAME'):
    with tf.variable_scope(name):
        with tf.name_scope('out_shape_calc'):
            dyn_input_shape = tf.shape(prev_layer)
            batch_size = dyn_input_shape[0]
            prev_channels = int(prev_layer.shape[3])

            if out_shape:
                output_shape = tf.stack([batch_size, out_shape[1], out_shape[2], num_filters])
            else:
                output_shape = tf.stack([batch_size, dyn_input_shape[1] * dim_upsampling_ratio, dyn_input_shape[2] * dim_upsampling_ratio, num_filters])
                
        if not std:
            std = np.sqrt(2 / prev_channels)
            
        weights = tf.get_variable('weights', (filter_size, filter_size, num_filters, prev_channels), initializer=tf.truncated_normal_initializer(stddev=std))
        biases = tf.get_variable('biases', [num_filters], initializer=tf.zeros_initializer())
        
        outputs = tf.nn.conv2d_transpose(prev_layer, weights, output_shape, strides=[1, stride, stride, 1], padding=padding, name='deconvolution') + biases
        layer = activate(outputs, activation)

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('outputs', outputs)
        tf.summary.histogram('activations', layer)

        layer = tf.reshape(layer, (-1, out_shape[1], out_shape[2], num_filters))

        return layer

def flatten(prev_layer, name='flatten'):
    with tf.name_scope(name):
        with tf.name_scope('out_shape_calc'):
            num_dimensions = int(np.prod(prev_layer.shape[1:]))

        return tf.reshape(prev_layer, (-1, num_dimensions))

def inflate(prev_layer, img_dimensions, name='inflate'):
    with tf.name_scope(name):
        with tf.name_scope('out_shape_calc'):
            num_channels = int(int(prev_layer.shape[1]) / np.prod(img_dimensions))

        return tf.reshape(prev_layer, (-1, img_dimensions[0], img_dimensions[1], num_channels))
    
def fc_stack(prev_layer, neuron_vector, activation_dict=None, name_vector=None):
    if activation_dict is None:
        activation_dict = {'None': []}
    
    activation_vector = dict_to_list(activation_dict, len(neuron_vector))
    
    for neurons, activation in zip(neuron_vector, activation_vector):
        layer = add_fc_layer(prev_layer, neurons, activation)
        prev_layer = layer
        
    return layer