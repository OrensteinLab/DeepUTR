import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers



###########################################################

def create_layer(layer_params, nodes_names_to_nodes, output_shape):
    dispatcher = load_layer_dispatcher()
    dispatcher[layer_params['layer type']](layer_params, nodes_names_to_nodes, output_shape)
    
def load_layer_dispatcher():
    dispatcher = {'cnn' : create_cnn_layer,
                  'gru' : create_gru_layer,
                  'global_max_pooling' : create_global_max_pooling_layer,
                  'max_pooling' : create_max_pooling_layer,
                  'dropout' : create_dropout_layer,
                  'SpatialDropout1D' : create_SpatialDropout1D_layer,
                  'concatenate' : create_concatenate_layer,
                  'dense' :create_dense_layer,
                  'embedding' : create_embedding_layer,
                  'BatchNormalization' : create_BatchNormalization_layer,
                  'Activation' : create_Activation_layer,
                  'flatten': create_flatten_layer
                 }
    return dispatcher


def create_regularizer (kernel_regularizer_type, kernel_regularizer_lambda):
    regularizer = None
    if (kernel_regularizer_type == 'l1'):
        regularizer = regularizers.l1(kernel_regularizer_lambda)
    elif (kernel_regularizer_type == 'l2'):
        regularizer = regularizers.l2(kernel_regularizer_lambda)
    elif (kernel_regularizer_type == 'l1_l2'):
        regularizer = regularizers.l1_l2(l1=kernel_regularizer_lambda[0], l2=kernel_regularizer_lambda[1])

    return regularizer

def create_cnn_layer(layer_params, nodes_names_to_nodes, output_shape):
        kernel_regularizer = create_regularizer (layer_params['kernel_regularizer_type'], layer_params['kernel_regularizer_lambda'])
        
        input_node = nodes_names_to_nodes[layer_params['input_names']]
        nodes_names_to_nodes[layer_params['output_names']] = layers.Conv1D(filters=layer_params['filters'],
                                                                    kernel_size=layer_params['kernel_size'],
                                                                    strides=layer_params['strides'],
                                                                    kernel_initializer=layer_params['kernel_initializer'],
                                                                    activation=layer_params['activation'],
                                                                    kernel_regularizer=kernel_regularizer,
                                                                    padding=layer_params['padding'],
                                                                    use_bias=layer_params['use_bias'],
                                                                    bias_initializer=layer_params['bias_initializer'])(input_node)
           
def create_gru_layer(layer_params, nodes_names_to_nodes, output_shape):
        kernel_regularizer = create_regularizer (layer_params['kernel_regularizer_type'], layer_params['kernel_regularizer_lambda'])
        
        input_node = nodes_names_to_nodes[layer_params['input_names']]
        nodes_names_to_nodes[layer_params['output_names']] = layers.Bidirectional(layers.GRU(units=layer_params['units'],
                                                                               activation=layer_params['activation'],
                                                                               use_bias=layer_params['use_bias'],
                                                                               kernel_initializer=layer_params['kernel_initializer'],
                                                                               bias_initializer=layer_params['bias_initializer'],
                                                                               kernel_regularizer=kernel_regularizer,
                                                                               return_sequences=layer_params['return_sequences']))(input_node)

def create_global_max_pooling_layer(layer_params, nodes_names_to_nodes, output_shape):
    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.GlobalMaxPooling1D()(input_node)
       
def create_max_pooling_layer(layer_params, nodes_names_to_nodes, output_shape):
    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.MaxPooling1D(pool_size=layer_params['pool_size'],
                                                                            strides=layer_params['strides'])(input_node)

def create_dropout_layer(layer_params, nodes_names_to_nodes, output_shape):
    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.Dropout(rate=layer_params['rate'])(input_node)

def create_concatenate_layer(layer_params, nodes_names_to_nodes, output_shape):
    inputs_names = layer_params['input_names']
    input_nodes = []
    for input_name in inputs_names:
        input_nodes.append(nodes_names_to_nodes[input_name])
    
    nodes_names_to_nodes[layer_params['output_names']] = layers.Concatenate()(input_nodes)
    
def create_dense_layer(layer_params, nodes_names_to_nodes, output_shape):
    kernel_regularizer = create_regularizer (layer_params['kernel_regularizer_type'], layer_params['kernel_regularizer_lambda'])
    bias_regularizer = create_regularizer (layer_params['bias_regularizer_type'], layer_params['bias_regularizer_type'])
    
    units = output_shape if layer_params['units']=='output shape' else layer_params['units']

    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.Dense(units=units,
                                                                     activation=layer_params['activation'],
                                                                     use_bias=layer_params['use_bias'],
                                                                     kernel_initializer=layer_params['kernel_initializer'],
                                                                     bias_initializer=layer_params['bias_initializer'],
                                                                     kernel_regularizer=kernel_regularizer,
                                                                     bias_regularizer=bias_regularizer)(input_node)

def create_embedding_layer (layer_params, nodes_names_to_nodes, output_shape):
    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.Embedding(input_dim=layer_params['input_dim'],
                                                                         output_dim=layer_params['output_dim'],
                                                                         input_length=layer_params['input_length'])(input_node)

def create_SpatialDropout1D_layer (layer_params, nodes_names_to_nodes, output_shape):
    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.SpatialDropout1D(rate=layer_params['rate'])(input_node)

def create_BatchNormalization_layer (layer_params, nodes_names_to_nodes, output_shape):
    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.BatchNormalization(axis=layer_params['axis'])(input_node)

def create_Activation_layer (layer_params, nodes_names_to_nodes, output_shape):
    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.Activation(layer_params['activation'])(input_node)

def create_flatten_layer (layer_params, nodes_names_to_nodes, output_shape):
    input_node = nodes_names_to_nodes[layer_params['input_names']]
    nodes_names_to_nodes[layer_params['output_names']]= layers.Flatten()(input_node)

###########################################################

    
def cnn_layer_params ( input_names,
                       output_names,
                       filters,
                       kernel_size,
                       strides=1,
                       padding="valid",
                       activation=None,
                       use_bias=True,
                       kernel_initializer="glorot_uniform",
                       bias_initializer="zeros",
                       kernel_regularizer_type=None, #options: 'l1', 'l2', 'l1_l2', None
                       kernel_regularizer_lambda = 0 #for l1 or l2: value, for l1_l2: (value_l1, vlauue_l2)
                      ):
    dictionary = vars()
    dictionary.update({'layer type': 'cnn'})
    return dictionary


def gru_layer_params ( input_names,
                       output_names,
                       units,
                       activation='tanh',
                       use_bias=True,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros',
                       kernel_regularizer_type=None, #options: 'l1', 'l2', 'l1_l2', None
                       kernel_regularizer_lambda=None, #for l1 or l2: value, for l1_l2: (value_l1, vlauue_l2) 
                       return_sequences=False
                      ):
    dictionary = vars()
    dictionary.update({'layer type': 'gru'})
    return dictionary


def global_max_pooling_layer_params (input_names,
                                     output_names
                                    ):
    dictionary = vars()
    dictionary.update({'layer type': 'global_max_pooling'})
    return dictionary


def max_pooling_layer_params (input_names,
                              output_names,
                              pool_size,
                              strides
                              ):
    dictionary = vars()
    dictionary.update({'layer type': 'max_pooling'})
    return dictionary


def flatten_layer_params (input_names,
                          output_names
                         ):
    dictionary = vars()
    dictionary.update({'layer type': 'flatten'})
    return dictionary


def dropout_layer_params (input_names,
                          output_names,
                          rate
                         ):
    dictionary = vars()
    dictionary.update({'layer type': 'dropout'})
    return dictionary


def concatenate_layer_params (input_names,
                              output_names
                             ):
    dictionary = vars()
    dictionary.update({'layer type': 'concatenate'})
    return dictionary


def dense_layer_params (input_names,
                        output_names,
                        units='output shape', #options: 'output shape' or integer
                        activation=None,
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        kernel_regularizer_type=None, #options: 'l1', 'l2', 'l1_l2', None
                        kernel_regularizer_lambda=None, #for l1 or l2: value, for l1_l2: (value_l1, vlauue_l2)
                        bias_regularizer_type=None, #options: 'l1', 'l2', 'l1_l2', None
                        bias_regularizer_lambda=None #for l1 or l2: value, for l1_l2: (value_l1, vlauue_l2) 
                        ):
    dictionary = vars()
    dictionary.update({'layer type': 'dense'})
    return dictionary

def embedding_layer_params (input_names,
                            output_names,
                            input_dim,
                            output_dim,
                            input_length=None
                            ):
    dictionary = vars()
    dictionary.update({'layer type': 'embedding'})
    return dictionary

def SpatialDropout1D_layer_params (input_names,
                                   output_names,
                                   rate
                                  ):
    dictionary = vars()
    dictionary.update({'layer type': 'SpatialDropout1D'})
    return dictionary

def BatchNormalization_layer_params (input_names,
                                     output_names,
                                     axis=-1
                                    ):
    dictionary = vars()
    dictionary.update({'layer type': 'BatchNormalization'})
    return dictionary

def Activation_layer_params (input_names,
                             output_names,
                             activation='relu'
                            ):
    dictionary = vars()
    dictionary.update({'layer type': 'Activation'})
    return dictionary



###########################################################

def load_model_params (model_id,
                       verbose=2,
                       epochs=128,
                       batch_size=32,
                       lr_scheduler=False,
                       tensor_board=False,
                       early_stooping_min_delta=0,
                       early_stooping_patience=10,
                       optimizer='Adam',
                       lr=0.0002,
                       loss='mse',
                       model_checkpoint=False,
                       gradient_clipping_norm=None
                      ):
    dictionary = vars()
    return dictionary