import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from deg_project.general import sequence_utilies
from deg_project.general import evaluation_utilies
import tensorflow as tf


def load_dataset (seq_path, labels_path_minus, labels_path_plus, data_type):
    seq = pd.read_csv(seq_path)["seq"].values.tolist()
    features_list = [sequence_utilies.one_hot_encoding(x) for x in seq]
    features = np.asarray(features_list, dtype="uint8")

    labels_minus_df = None
    labels_plus_df = None
    if(data_type=='-' or data_type=='-+'):
        labels_minus_df = pd.read_csv(labels_path_minus)
    if(data_type =='+' or data_type=='-+'):
        labels_plus_df= pd.read_csv(labels_path_plus) 
        
    return features, labels_minus_df, labels_plus_df
###################################################################################
# load dataset_model_8_points - for validation_seq- split=False, for train_validation_test - split=True
def load_dataset_model_8_points (seq_path, labels_path_minus, labels_path_plus, data_type, split, index_for_split=None):
    features, labels_minus_df, labels_plus_df= load_dataset (seq_path, labels_path_minus, labels_path_plus, data_type)
    
    if(data_type == '-+'):
        initial_values_features_minus, labels_minus = split_mRNA_levels_to_initial_and_labels(labels_minus_df)
        initial_values_features_plus, labels_plus = split_mRNA_levels_to_initial_and_labels(labels_plus_df)
        initial_values_features = np.concatenate((initial_values_features_minus, initial_values_features_plus), axis=1)
        labels = np.concatenate((labels_minus, labels_plus), axis=1)
    elif(data_type == '-' or data_type == '+'):
        labels_df = labels_minus_df if data_type == '-' else labels_plus_df
        initial_values_features, labels = split_mRNA_levels_to_initial_and_labels(labels_df)
    else:
        raise ValueError('invalid data type')
    
    #initial return_value with dataset without splitting
    return_value = (initial_values_features, features, labels)
    if(split == True):
        return_value = split_features_and_labels_for_8_points (features, initial_values_features, labels, index_for_split)
       
    return return_value

def split_mRNA_levels_to_initial_and_labels(labels_df):
    initial_values_features =  labels_df.iloc[:, 1].values
    initial_values_features = initial_values_features.reshape(initial_values_features.shape[0],1)
    labels =  labels_df.iloc[:, 2:].values

    return initial_values_features, labels

def split_features_and_labels_for_8_points (features, initial_values_features, labels, index_for_split):
        if(index_for_split is None):
            train_x, test_and_validation_x, train_x_initial, test_and_validation_x_initial, train_y, test_and_validation_y = train_test_split(features, initial_values_features, labels, test_size=0.22, random_state=31)
            test_x, validation_x, test_x_initial, validation_x_initial, test_y, validation_y = train_test_split(test_and_validation_x, test_and_validation_x_initial, test_and_validation_y, test_size=0.5, random_state=31)
            return_value = (train_x_initial, train_x, train_y),  (validation_x_initial, validation_x, validation_y), (test_x_initial, test_x, test_y)
        else:
            df = pd.read_csv(index_for_split)
            train_ids = df['train_ids'].dropna().astype(int).values
            validation_ids = df['validation_ids'].dropna().astype(int).values
            test_ids = df['test_ids'].dropna().astype(int).values
            return_value = (initial_values_features[train_ids], features[train_ids], labels[train_ids]),  (initial_values_features[validation_ids], features[validation_ids], labels[validation_ids]), (initial_values_features[test_ids], features[test_ids], labels[test_ids])

        return return_value


###################################################################################
# load dataset_linear_model - for validation_seq- split=False, for train_validation_test - split=True
def load_dataset_linear_model (seq_path, labels_path_minus, labels_path_plus, data_type, split, index_for_split=None):
    features, labels_minus_df, labels_plus_df = load_dataset(seq_path, labels_path_minus, labels_path_plus, data_type)
    
    if(data_type == '-+'):
        labels_minus =  labels_minus_df.iloc[:, 1:].values
        labels_plus =  labels_plus_df.iloc[:, 1:].values
        labels_minus = evaluation_utilies.compute_LR_slopes(labels_minus)
        labels_plus = evaluation_utilies.compute_LR_slopes(labels_plus)
        labels = np.concatenate((np.expand_dims(labels_minus, axis=1), np.expand_dims(labels_plus, axis=1)), axis=1)
    elif (data_type == '-' or data_type == '+'):
        labels_df = labels_minus_df if data_type == '-' else labels_plus_df
        labels =  evaluation_utilies.compute_LR_slopes(labels_df.iloc[:, 1:].values) 
    else:
        raise ValueError('invalid data type')
    
    #initial return_value with dataset without splitting
    return_value = (features, labels)
    if(split == True):
        return_value = split_features_and_labels_for_linear (features, labels, index_for_split)
    
    return return_value

def split_features_and_labels_for_linear (features, labels, index_for_split):
        if(index_for_split is None):
            train_x, test_and_validation_x, train_y, test_and_validation_y = train_test_split(features, labels, test_size=0.22, random_state=31)
            test_x, validation_x, test_y, validation_y = train_test_split(test_and_validation_x, test_and_validation_y, test_size=0.5, random_state=31)
            return_value = (train_x, train_y),  (validation_x, validation_y), (test_x, test_y)
        else:
            df = pd.read_csv(index_for_split)
            train_ids = df['train_ids'].dropna().astype(int).values
            validation_ids = df['validation_ids'].dropna().astype(int).values
            test_ids = df['test_ids'].dropna().astype(int).values
            return_value = (features[train_ids], labels[train_ids]), (features[validation_ids], labels[validation_ids]), (features[test_ids], labels[test_ids])

        return return_value

###################################################################################
# load dataset_linear_model - for validation_seq- split=False, for train_validation_test - split=True
def load_dataset_model_type (seq_path, labels_path_minus, labels_path_plus, model_type, data_type, split, index_for_split=None):

    if (model_type == 'dynamics'):
        return load_dataset_model_8_points (seq_path, labels_path_minus, labels_path_plus, data_type, split, index_for_split)
    elif (model_type == 'rate'):
        return load_dataset_linear_model (seq_path, labels_path_minus, labels_path_plus, data_type, split, index_for_split)
    else:
        raise ValueError('invalid model type')
