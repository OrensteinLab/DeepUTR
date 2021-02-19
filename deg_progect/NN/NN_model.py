import pandas as pd
from pathlib import Path
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from deg_project.NN import NN_utilies
from deg_project.NN import NN_layers
from deg_project.general import general_utilies

def scheduler(epoch, lr):
    #don't change the the lr for the first 10 epochs, and then change after 3 epochs.
    if epoch < 10 or epoch%3 != 1: 
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def model_creation_and_fit (train_set_wrapper,  validation_set_wrapper, model_type, data_type, layers_params, model_params, model_id_and_timestamp):
   
    # Clear clutter from previous Keras session graphs.
    # Clear only when not running multiple models in parallel
    clear_session()
    
    #create inputs layers
    nodes_names_to_nodes = {}
    train_x, train_y, validation_x, validation_y = None, None, None, None
    inputs = None
    if (model_type=='linear'):
        #unwrap tarin and validation set
        (train_x, train_y) = train_set_wrapper
        (validation_x, validation_y) = validation_set_wrapper
        #create input layer
        nodes_names_to_nodes.update({'input1' : layers.Input(shape=(train_x.shape[1], train_x.shape[2]))}) #layers.Input(shape=(None, train_x.shape[2]))})
        inputs = [nodes_names_to_nodes['input1']]
    elif (model_type=='multi_task_model_8_points' or model_type=='A_minus_model_8_points' or  model_type=='A_plus_model_8_points'):
        #unwrap tarin and validation set
        (train_x_intial, train_x, train_y) = train_set_wrapper
        (validation_x_intial, validation_x, validation_y) = validation_set_wrapper
        #create input layers
        nodes_names_to_nodes.update({'input1' : layers.Input(shape=(train_x.shape[1], train_x.shape[2]))})
        nodes_names_to_nodes.update({'input2' : layers.Input(shape=(train_x_intial.shape[1],))})
        inputs = [nodes_names_to_nodes['input1'], nodes_names_to_nodes['input2']]
        #combine intial and one-hot to list for future use
        train_x = [train_x, train_x_intial]
        validation_x =[validation_x, validation_x_intial]
    else:
        raise ValueError('invalid model type')

    if (model_type == 'linear'):
        if(data_type == "A_minus_and_plus"):
            output_shape = 2
        else:
            output_shape = 1
    else:
        output_shape = train_y.shape[1]
    
    #create layers
    for layer_params in layers_params:
        NN_layers.create_layer(layer_params, nodes_names_to_nodes, output_shape)
        
    #define inputs-outputs and complie model
    model = Model(inputs=inputs, outputs=nodes_names_to_nodes['output'])

    #print model description summary
    print (model.summary())    
    
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=2e-4, decay_steps=10000, decay_rate=0.9)
    if (model_params['gradient_clipping_norm'] is not None):
        optimizer = tf.keras.optimizers.get({"class_name": model_params['optimizer'], "config": {"learning_rate": model_params['lr'], "clipnorm": model_params['gradient_clipping_norm'] }})
    else:
        optimizer = tf.keras.optimizers.get({"class_name": model_params['optimizer'], "config": {"learning_rate": model_params['lr']}})
    
    model.compile(loss=model_params['loss'], optimizer=optimizer, metrics=['mse', NN_utilies.tf_pearson, 'mae'])
    
    #create callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    if(model_params['lr_scheduler'] == True):
       callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
    if (model_params['tensor_board'] == True):
        logdir = os.path.join(general_utilies.files_dir+"logs", model_type+model_id_and_timestamp)
        callbacks.append(tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1))
    if (model_params['early_stooping_patience']>0):
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_params['early_stooping_patience'], min_delta=model_params['early_stooping_min_delta'],  verbose=1, restore_best_weights=True))
    if(model_params['model_checkpoint'] == True):
        dir_path = general_utilies.files_dir+"models_chekpoints/"+model_type+model_id_and_timestamp
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        filepath = dir_path+"/"+"saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1))

    #fit
    model.fit(train_x,
              train_y,
              validation_data=(validation_x, validation_y),
              shuffle=True,
              epochs=model_params['epochs'],                             
              batch_size= model_params['batch_size'],
              verbose=model_params['verbose'],
              callbacks=callbacks)
    
    return model



