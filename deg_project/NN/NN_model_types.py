from deg_project.NN import NN_layers

def create_model_spec(model_id, train_spec=None, layers_spec=None):
    model_params, layers_params = globals()[model_id](model_id, train_spec, layers_spec)

    return model_params, layers_params

def model_8_points_id_1 (model_id, train_spec, layers_spec):
    model_params = NN_layers.load_model_params (model_id=model_id)
    if(train_spec is not None):
        model_params.update(train_spec)

    layers_spec_default = {'filters' : 512, 'kernel_size' : 10, 'CNN_activation' : 'relu', 
                           'CNN_kernel_and_bias_initializer' : 'RandomNormal',
                           'CNN_kernel_regularizer_lambda' : 5e-3, 'dense_units' : [512], 'dense_activation' : 'relu' }

    layers_spec_updated = layers_spec_default
    if(layers_spec is not None):
         layers_spec_updated.update(layers_spec)

    layers_params =[]
    layers_params.append(NN_layers.cnn_layer_params(input_names='input1',
                                                    output_names='x11',
                                                    filters=layers_spec_updated['filters'],
                                                    kernel_size=layers_spec_updated['kernel_size'],
                                                    strides=1,
                                                    padding='same',
                                                    activation=layers_spec_updated['CNN_activation'],
                                                    use_bias=True,
                                                    kernel_initializer=layers_spec_updated['CNN_kernel_and_bias_initializer'],
                                                    bias_initializer=layers_spec_updated['CNN_kernel_and_bias_initializer'],
                                                    kernel_regularizer_type='l2',
                                                    kernel_regularizer_lambda=layers_spec_updated['CNN_kernel_regularizer_lambda']))

    layers_params.append(NN_layers.global_max_pooling_layer_params(input_names='x11',
                                                                   output_names='x12'))

    layers_params.append(NN_layers.concatenate_layer_params(input_names=['x12', 'input2'],
                                                            output_names='x31'))

    for i in range(len(layers_spec_updated['dense_units'])):
        input_names, output_names = 'x', 'x'
        if(i == 0):
            input_names = 'x31'
        if (i == len(layers_spec_updated['dense_units'])-1):
            output_names = 'x32'
        layers_params.append(NN_layers.dense_layer_params(input_names=input_names,
                                                          output_names=output_names,
                                                          units=layers_spec_updated['dense_units'][i],
                                                          activation=layers_spec_updated['dense_activation']))

    layers_params.append(NN_layers.dense_layer_params(input_names='x32',
                                                      output_names='output',
                                                      units='output shape',
                                                      activation='linear'))

    return model_params, layers_params

def model_8_points_id_2 (model_id, train_spec=None, layers_spec=None):
    model_params = NN_layers.load_model_params (model_id=model_id)
    if(train_spec is not None):
        model_params.update(train_spec)

    layers_spec_default = {'GRU_units' : 256, 'GRU_activation' : 'tanh', 'dropout_rate' : 0.2,
                           'dense_units' : [128], 'dense_activation' : 'relu' }

    layers_spec_updated = layers_spec_default
    if(layers_spec is not None):
         layers_spec_updated.update(layers_spec)

    layers_params =[]
    layers_params.append(NN_layers.gru_layer_params(input_names='input1',
                                                    output_names='x11',
                                                    units=layers_spec_updated['GRU_units'],
                                                    activation=layers_spec_updated['GRU_activation']))

    layers_params.append(NN_layers.dropout_layer_params(input_names='x11',
                                                        output_names='x12',
                                                        rate=layers_spec_updated['dropout_rate']))

    layers_params.append(NN_layers.concatenate_layer_params(input_names=['x12', 'input2'],
                                                            output_names='x31'))
    
    for i in range(len(layers_spec_updated['dense_units'])):
        input_names, output_names = 'x', 'x'
        if(i == 0):
            input_names = 'x31'
        if (i == len(layers_spec_updated['dense_units'])-1):
            output_names = 'x32'
        layers_params.append(NN_layers.dense_layer_params(input_names=input_names,
                                                          output_names=output_names,
                                                          units=layers_spec_updated['dense_units'][i],
                                                          activation=layers_spec_updated['dense_activation']))

    layers_params.append(NN_layers.dense_layer_params(input_names='x32',
                                                      output_names='output',
                                                      units='output shape',
                                                      activation='linear'))
    return model_params, layers_params

def linear_model_points_id_1 (model_id, train_spec, layers_spec):
    model_params = NN_layers.load_model_params (model_id=model_id)
    if(train_spec is not None):
        model_params.update(train_spec)

    layers_spec_default = {'filters' : 512, 'kernel_size' : 10, 'CNN_activation' : 'relu', 
                           'CNN_kernel_and_bias_initializer' : 'RandomNormal',
                           'CNN_kernel_regularizer_lambda' : 5e-3, 'dense_units' : [512], 'dense_activation' : 'relu' }

    layers_spec_updated = layers_spec_default
    if(layers_spec is not None):
         layers_spec_updated.update(layers_spec)

    layers_params =[]
    layers_params.append(NN_layers.cnn_layer_params(input_names='input1',
                                                    output_names='x11',
                                                    filters=layers_spec_updated['filters'],
                                                    kernel_size=layers_spec_updated['kernel_size'],
                                                    strides=1,
                                                    padding='same',
                                                    activation=layers_spec_updated['CNN_activation'],
                                                    use_bias=True,
                                                    kernel_initializer=layers_spec_updated['CNN_kernel_and_bias_initializer'],
                                                    bias_initializer=layers_spec_updated['CNN_kernel_and_bias_initializer'],
                                                    kernel_regularizer_type='l2',
                                                    kernel_regularizer_lambda=layers_spec_updated['CNN_kernel_regularizer_lambda']))

    layers_params.append(NN_layers.global_max_pooling_layer_params(input_names='x11',
                                                                   output_names='x12'))

    for i in range(len(layers_spec_updated['dense_units'])):
        input_names, output_names = 'x', 'x'
        if(i == 0):
            input_names = 'x12'
        if (i == len(layers_spec_updated['dense_units'])-1):
            output_names = 'x13'
        layers_params.append(NN_layers.dense_layer_params(input_names=input_names,
                                                          output_names=output_names,
                                                          units=layers_spec_updated['dense_units'][i],
                                                          activation=layers_spec_updated['dense_activation']))

    layers_params.append(NN_layers.dense_layer_params(input_names='x13',
                                                      output_names='output',
                                                      units='output shape',
                                                      activation='linear'))

    return model_params, layers_params

def linear_model_points_id_2 (model_id, train_spec=None, layers_spec=None):
    model_params = NN_layers.load_model_params (model_id=model_id)
    if(train_spec is not None):
        model_params.update(train_spec)

    layers_spec_default = {'GRU_units' : 256, 'GRU_activation' : 'tanh', 'dropout_rate' : 0.2,
                            'dense_units' : [128], 'dense_activation' : 'relu' }

    layers_spec_updated = layers_spec_default
    if(layers_spec is not None):
            layers_spec_updated.update(layers_spec)

    layers_params =[]
    layers_params.append(NN_layers.gru_layer_params(input_names='input1',
                                                    output_names='x11',
                                                    units=layers_spec_updated['GRU_units'],
                                                    activation=layers_spec_updated['GRU_activation']))

    layers_params.append(NN_layers.dropout_layer_params(input_names='x11',
                                                        output_names='x12',
                                                        rate=layers_spec_updated['dropout_rate']))

    for i in range(len(layers_spec_updated['dense_units'])):
        input_names, output_names = 'x', 'x'
        if(i == 0):
            input_names = 'x12'
        if (i == len(layers_spec_updated['dense_units'])-1):
            output_names = 'x13'
        layers_params.append(NN_layers.dense_layer_params(input_names=input_names,
                                                            output_names=output_names,
                                                            units=layers_spec_updated['dense_units'][i],
                                                            activation=layers_spec_updated['dense_activation']))

    layers_params.append(NN_layers.dense_layer_params(input_names='x13',
                                                        output_names='output',
                                                        units='output shape',
                                                        activation='linear'))
    return model_params, layers_params

