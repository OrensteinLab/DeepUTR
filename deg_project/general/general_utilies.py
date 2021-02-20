import pandas as pd
import numpy as np
import pickle
import pathlib

from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import stats

from sklearn.linear_model import LinearRegression

import statistics


###############paths######################
home_dir = str(pathlib.Path(__file__).parent.parent.parent.absolute())+"/" #deg_project path
files_dir = home_dir+"files/"

seq_PATH = files_dir+"dataset/mRNA_sequences.csv"
A_minus_normalized_levels_PATH = files_dir+"dataset/A_minus_normalized_levels.csv"
A_plus_normalized_levels_PATH = files_dir+"dataset/A_plus_normalized_levels.csv"
validation_seq_PATH = files_dir+"dataset/validation_seq.csv"
validation_A_minus_normalized_levels_PATH = files_dir+"dataset/validation_A_minus_normalized_levels.csv"
validation_A_plus_normalized_levels_PATH = files_dir+"dataset/validation_A_plus_normalized_levels.csv"
features_1_7_kmers_PATH = files_dir+"dataset/features_1-7kmers.sav"
features_1_8_kmers_PATH = files_dir+"dataset/RF_features_1-8kmers.sav"



RESA_data_PATH = files_dir+"dataset/RESA_WindowSequences.csv"

split_to_train_validation_test_disjoint_sets_ids_PATH = files_dir+'dataset/split_to_train_validation_test_disjoint_sets_ids.csv'


#######################evalute pearson and MSE all 8 points models###############################################
#TODO: support linear model + reduce code 
from deg_project.lasso_RF import RF_and_lasso_train_test_models_utilies
from deg_project.NN import NN_train_test_models_utilies
def evalute_pearson_and_RMSE_all_8_points_models(saved_models_path_list, model_type_list,
                                                 model_id_list, data_type_list, saved_models_path_list_RF_lasso,
                                                 validation_seq_or_test, index_for_split=None):
    pearson_values_dict = {}
    RMSE_values_dict = {}
    ########load pearson values of the NN models##################
    for i in range(len(model_type_list)):
        NN_metrics_results = NN_train_test_models_utilies.evaluate_model_type (model_path=saved_models_path_list[i], seq_path=seq_PATH, labels_path_minus=A_minus_normalized_levels_PATH,
                                                                                   labels_path_plus=A_plus_normalized_levels_PATH, validate_seq_path=validation_seq_PATH,
                                                                                   validate_labels_path_minus=validation_A_minus_normalized_levels_PATH,
                                                                                   validate_labels_path_plus=validation_A_plus_normalized_levels_PATH, model_id = model_id_list[i],
                                                                                   model_type=model_type_list[i], data_type=data_type_list[i], compute_mean_and_std=False, index_for_split=index_for_split)

        if (data_type_list[i] == 'A_minus_and_plus'):
            pearson_values_dict[model_id_list[i]+'_multi_A_minus'] = NN_metrics_results[validation_seq_or_test][0]['pearson']
            pearson_values_dict[model_id_list[i]+'_multi_A_plus'] = NN_metrics_results[validation_seq_or_test][1]['pearson']
            RMSE_values_dict[model_id_list[i]+'_multi_A_minus'] = NN_metrics_results[validation_seq_or_test][0]['RMSE']
            RMSE_values_dict[model_id_list[i]+'_multi_A_plus'] = NN_metrics_results[validation_seq_or_test][1]['RMSE']
        else:
            pearson_values_dict[model_id_list[i]+'_'+data_type_list[i]] = NN_metrics_results[validation_seq_or_test][0]['pearson']
            RMSE_values_dict[model_id_list[i]+'_'+data_type_list[i]] = NN_metrics_results[validation_seq_or_test][0]['RMSE']
    ########################################################################################

    ########################load pearson values of the RF and lasso models##################

    lasso_metrics_results = RF_and_lasso_train_test_models_utilies.evaluate_lasso_or_RF_model (model_A_minus_path=saved_models_path_list_RF_lasso[0], model_A_plus_path=saved_models_path_list_RF_lasso[1], seq_path=seq_PATH,
                                                                                            labels_path_minus=A_minus_normalized_levels_PATH, labels_path_plus=A_plus_normalized_levels_PATH,
                                                                                            validate_seq_path=validation_seq_PATH, validate_labels_path_minus=validation_A_minus_normalized_levels_PATH,
                                                                                            validate_labels_path_plus=validation_A_plus_normalized_levels_PATH, model_type='model_8_points', lasso_or_RF='lasso',
                                                                                            features_path=features_1_7_kmers_PATH, compute_mean_and_std=False, index_for_split=index_for_split)

    pearson_values_dict['lasso_A_minus'] = lasso_metrics_results[0][validation_seq_or_test][0]['pearson']
    pearson_values_dict['lasso_A_plus'] = lasso_metrics_results[1][validation_seq_or_test][0]['pearson']
    RMSE_values_dict['lasso_A_minus'] = lasso_metrics_results[0][validation_seq_or_test][0]['RMSE']
    RMSE_values_dict['lasso_A_plus'] = lasso_metrics_results[1][validation_seq_or_test][0]['RMSE']

    
    RF_metrics_results = RF_and_lasso_train_test_models_utilies.evaluate_lasso_or_RF_model (model_A_minus_path=saved_models_path_list_RF_lasso[2], model_A_plus_path=saved_models_path_list_RF_lasso[3], seq_path=seq_PATH,
                                                                                            labels_path_minus=A_minus_normalized_levels_PATH, labels_path_plus=A_plus_normalized_levels_PATH,
                                                                                            validate_seq_path=validation_seq_PATH, validate_labels_path_minus=validation_A_minus_normalized_levels_PATH,
                                                                                            validate_labels_path_plus=validation_A_plus_normalized_levels_PATH, model_type='model_8_points', lasso_or_RF='RF',
                                                                                            features_path=features_1_8_kmers_PATH, compute_mean_and_std=False, index_for_split=index_for_split)

    pearson_values_dict['RF_A_minus'] = RF_metrics_results[0][validation_seq_or_test][0]['pearson']
    pearson_values_dict['RF_A_plus'] = RF_metrics_results[1][validation_seq_or_test][0]['pearson']
    RMSE_values_dict['RF_A_minus'] = RF_metrics_results[0][validation_seq_or_test][0]['RMSE']
    RMSE_values_dict['RF_A_plus'] = RF_metrics_results[1][validation_seq_or_test][0]['RMSE']


    return pearson_values_dict, RMSE_values_dict



###############################lock using lock file######################################
import fcntl

def acquireLock():
    ''' acquire exclusive lock file access '''
    locked_file_descriptor = open('lockfile.LOCK', 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor

def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()
