import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

from scipy import sparse
from scipy.sparse import csr_matrix

from deg_project.general import sequence_utilies 
from deg_project.general import evaluation_utilies
from deg_project.general import general_utilies


def load_dataset (seq_path,  labels_path_minus, labels_path_plus, lasso_or_RF, features_path):
    if (lasso_or_RF == 'lasso'):
        min_kmer_length = 3
        max_kmer_length = 7
    else:
        max_kmer_length = 8
        min_kmer_length = 1
        
    if(features_path == None):
        allKmers = sequence_utilies.retutnAllKmers(min_kmer_length, max_kmer_length)
        sequences = pd.read_csv(seq_path)["seq"].values.tolist()
        features = sequence_utilies.createFeturesVectorsForAllSeq (allKmers=allKmers, min_kmer_length=min_kmer_length, max_kmer_length=max_kmer_length, sequences=sequences)
        features = features.reshape(features.shape[0],features.shape[1])
        features = csr_matrix(features, dtype=np.int8)
    else:
        features = sequence_utilies.pickle.load(open(features_path, 'rb'))
        if lasso_or_RF=='lasso':
            features = features[:,20:] # take 3-mer to 7-mer if the model is lasso
        
    labels_minus_df = pd.read_csv(labels_path_minus)
    labels_plus_df = pd.read_csv(labels_path_plus)

    return features, labels_minus_df, labels_plus_df

# load train and test data and prepare model input function - initial value is feature
def load_dataset_and_split (seq_path,  labels_path_minus, labels_path_plus, model_type , lasso_or_RF,features_path=None, index_for_split=None):
    features, labels_minus_df, labels_plus_df = load_dataset (seq_path,  labels_path_minus, labels_path_plus, lasso_or_RF,features_path)
    
    train_set_wrapper_minus, test_set_wrapper_minus =  split_to_train_test(features, labels_minus_df, model_type, index_for_split)
    train_set_wrapper_plus, test_set_wrapper_plus =  split_to_train_test(features, labels_plus_df, model_type, index_for_split)
    
    return train_set_wrapper_minus, test_set_wrapper_minus, train_set_wrapper_plus, test_set_wrapper_plus

# load validation - and initial value is feature
def load_validation_seq_dataset (seq_path,  labels_path_minus, labels_path_plus, model_type, lasso_or_RF ,features_path=None):
    features, labels_minus_df, labels_plus_df = load_dataset (seq_path,  labels_path_minus, labels_path_plus, lasso_or_RF,features_path)
    
    if(model_type == 'model_8_points'):
        features_minus, labels_minus = add_inital_value_to_features(features, labels_minus_df)
        features_plus, labels_plus = add_inital_value_to_features(features, labels_plus_df)
    elif(model_type =='linear'):
        features_minus, labels_minus = features, evaluation_utilies.compute_LR_slopes(labels_minus_df.iloc[:, 1:].values)
        features_plus, labels_plus =  features, evaluation_utilies.compute_LR_slopes(labels_plus_df.iloc[:, 1:].values)
    else:
        raise ValueError('invalid model_type') 

    return (features_minus, labels_minus), (features_plus, labels_plus)

def split_to_train_test(features, labels_df, model_type, index_for_split):
    if(model_type == 'model_8_points'):
        features, labels = add_inital_value_to_features(features, labels_df)
    elif(model_type == 'linear'):
        labels = evaluation_utilies.compute_LR_slopes(labels_df.iloc[:, 1:].values)
    else:
        raise ValueError('invalid model_type') 
        
    if (index_for_split is None):
        train_x, test_and_validation_x, train_y, test_and_validation_y = train_test_split(features, labels, test_size=0.22, random_state=31)
        test_x, validation_x, test_y, validation_y = train_test_split(test_and_validation_x, test_and_validation_y, test_size=0.5, random_state=31)
    else:
        df = pd.read_csv(index_for_split)
        train_ids = df['train_ids'].dropna().astype(int).values
        validation_ids = df['validation_ids'].dropna().astype(int).values
        test_ids = df['test_ids'].dropna().astype(int).values
        train_x, train_y = features[train_ids], labels[train_ids]
        test_x, test_y = features[test_ids], labels[test_ids]
        validation_x, validation_y = features[validation_ids], labels[validation_ids]
    
    #the two steps below are done in order to create train set that is the same as the train and validation set that were made for the NN models
    train_x = sparse.hstack((train_x.T, validation_x.T), format="csr").T
    train_y = np.concatenate((train_y, validation_y))
    
    return (train_x, train_y), (test_x, test_y)

def add_inital_value_to_features(features, labels_df):
    intial_values_features =  labels_df.iloc[:, 1].values
    intial_values_features = intial_values_features.reshape(intial_values_features.shape[0],1)

    features_with_intial_value = sparse.hstack((features,intial_values_features), format="csr")
    labels =  labels_df.iloc[:, 2:].values
    return features_with_intial_value, labels


def evluate_test_and_validation_seq (model, model_type,  test_set_wrapper, validation_seq_wrapper, compute_mean_and_std=True):
    results_dict = {}
    print('\033[1m' + "test evaluation" + '\033[0m')
    (test_x, test_y) = test_set_wrapper
    predicted_test =  model.predict(test_x)

    if(model_type == 'linear'):
        results_dict['test'] = evaluation_utilies.slope_test(predicted_test, test_y, compute_mean_and_std)
    else:
        results_dict['test'] = evaluation_utilies.evaluate_model (predicted_test, test_y, compute_mean_and_std)

    print('\033[1m' + "validation_seq evaluation" + '\033[0m')
    (validation_seq_features, validation_seq_labels) = validation_seq_wrapper
    predicted_validation_seq =  model.predict(validation_seq_features)


    if(model_type == 'linear'):
        results_dict['validation_seq'] = evaluation_utilies.slope_test(predicted_validation_seq, validation_seq_labels, compute_mean_and_std)
    else:
        results_dict['validation_seq'] = evaluation_utilies.evaluate_model (predicted_validation_seq, validation_seq_labels, compute_mean_and_std)

    return results_dict

def train_and_evluate(train_set_wrapper, test_set_wrapper, validation_seq_wrapper, model_type, lasso_or_RF):
    if (lasso_or_RF == "lasso"):
        model = lasso_model_creation_and_fit (train_set_wrapper)
    else:
        model = RF_model_creation_and_fit (train_set_wrapper)
    
    evluate_test_and_validation_seq (model, model_type,  test_set_wrapper, validation_seq_wrapper)

    return model


def RF_model_creation_and_fit (train_set_wrapper):
    (features, labels) = train_set_wrapper
    model = RandomForestRegressor(n_estimators = 500, random_state=0, verbose=2, n_jobs=-1)
    model.fit(features, labels)
    
    return model

def lasso_model_creation_and_fit (train_set_wrapper):
    (features, labels) = train_set_wrapper
    model= Lasso(alpha=0.001, max_iter=1000)
    model.fit(features, labels)
    
    return model


def train_test_validate_lasso_or_RF_model (seq_path, labels_path_minus, labels_path_plus, validate_seq_path, validate_labels_path_minus, validate_labels_path_plus, model_type, lasso_or_RF, features_path=None, index_for_split=None):
    train_set_wrapper_minus, test_set_wrapper_minus, train_set_wrapper_plus, test_set_wrapper_plus = load_dataset_and_split (seq_path,  labels_path_minus, labels_path_plus, model_type,  lasso_or_RF, features_path, index_for_split=index_for_split)
    
    validation_seq_wrapper_minus, validation_seq_wrapper_plus = load_validation_seq_dataset (validate_seq_path, validate_labels_path_minus, validate_labels_path_plus, model_type, lasso_or_RF, None) 
    
    print("##############A-###############")
    model_minus = train_and_evluate(train_set_wrapper_minus, test_set_wrapper_minus, validation_seq_wrapper_minus, model_type, lasso_or_RF)
    
    print("##############A+###############")
    model_plus = train_and_evluate(train_set_wrapper_plus, test_set_wrapper_plus, validation_seq_wrapper_plus, model_type, lasso_or_RF)
    
    return model_minus, test_set_wrapper_minus, validation_seq_wrapper_minus, model_plus, test_set_wrapper_plus, validation_seq_wrapper_plus


def evaluate_lasso_or_RF_model (model_A_minus_path, model_A_plus_path, seq_path, labels_path_minus, labels_path_plus, validate_seq_path, validate_labels_path_minus, validate_labels_path_plus, model_type, lasso_or_RF, features_path=None, compute_mean_and_std=True, index_for_split=None):
    _train_set_wrapper_minus, test_set_wrapper_minus, _train_set_wrapper_plus, test_set_wrapper_plus = load_dataset_and_split (seq_path,  labels_path_minus, labels_path_plus, model_type,  lasso_or_RF, features_path, index_for_split=index_for_split)

    validation_seq_wrapper_minus, validation_seq_wrapper_plus = load_validation_seq_dataset (validate_seq_path, validate_labels_path_minus, validate_labels_path_plus, model_type, lasso_or_RF, None) 
    
    results = []
    print("##############A-###############")
    model = general_utilies.pickle.load(open(model_A_minus_path, 'rb'))
    results.append(evluate_test_and_validation_seq (model, model_type, test_set_wrapper_minus, validation_seq_wrapper_minus, compute_mean_and_std))
    
    print("##############A+###############")
    model = general_utilies.pickle.load(open(model_A_plus_path, 'rb'))
    results.append(evluate_test_and_validation_seq (model, model_type, test_set_wrapper_plus, validation_seq_wrapper_plus, compute_mean_and_std))

    return results
    
    
  ###############################################################################
from scipy import stats
def evaluate_RESA_with_linear_model (model_path, type):
    #load model
    model = general_utilies.pickle.load(open(model_path, 'rb'))
    
    features, observed_values = load_RESA_data(type)

    if(type=='lasso'):
        preticted =  model.predict(features.reshape(25043, 21824))
    else:
        preticted =  model.predict(features.reshape(25043, 87380))
    
    print("(pearson, p-value):",stats.pearsonr(preticted.reshape((25043, )), observed_values))


def load_RESA_data(type):
    if(type=='lasso'):
        starting_index=3
        ending_index=7
    else:
        starting_index=1
        ending_index=8

    df = pd.read_csv  (general_utilies.RESA_data_PATH, sep="\t")
    df = df.drop(6377)
    df = df.reset_index()
    seq_df = df ["WindowSeq"]
    observed_values = np.array(df ["WindowRESA"])
    #create features for the model
    allKmers = sequence_utilies.retutnAllKmers (starting_index,ending_index)
    features = sequence_utilies.createFeturesVectorsForAllSeq (allKmers, starting_index, ending_index, seq_df)

    return features, observed_values