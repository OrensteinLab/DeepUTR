import tensorflow as tf
import datetime
import numpy as np
from scipy import stats

from deg_project.NN import NN_load_datasets
from deg_project.NN import NN_model
from deg_project.NN import NN_utilies
from deg_project.NN import NN_model_types


from deg_project.general import general_utilies
from deg_project.general import evaluation_utilies
from deg_project.NN import TF_modisco


###################################################################################
############################train and test utilitis################################
def predict_and_evaluate(model, set_wrapper, model_type, data_type, results, compute_mean_and_std=True, only_predict=False, only_evaluate=None):
    #linear model need different treatment 
    if (model_type == 'linear'):
        return linear_model_predict_and_evaluate (model, set_wrapper, model_type, data_type, results, compute_mean_and_std=True, only_predict=only_predict, only_evaluate=only_evaluate)

    #predict
    if(model_type == 'multi_task_model_8_points' or model_type == 'A_minus_model_8_points' or model_type == 'A_plus_model_8_points'):
        output_size = 8
        (x_initial, x, y) = set_wrapper
        predicted_test =  model.predict([x, x_initial]) if only_evaluate is None else only_evaluate
    
    if (only_predict):
        return predicted_test

    #evaluate
    if(model_type == 'multi_task_model_8_points'):
        print("A-")
        mertric_results = evaluation_utilies.evaluate_model (predicted_test[:,:output_size], y[:,:output_size], compute_mean_and_std)
        results = results + mertric_results
        print("A+")
        mertric_results = evaluation_utilies.evaluate_model (predicted_test[:,output_size:], y[:,output_size:], compute_mean_and_std)
        results = results + mertric_results
    else:
        mertric_results = evaluation_utilies.evaluate_model (predicted_test, y, compute_mean_and_std)
        if (data_type=='A_minus'):
            results = results + mertric_results + [None,None,None]  if compute_mean_and_std else results + mertric_results
        else:
            results = results + [None,None,None] + mertric_results if compute_mean_and_std else results + mertric_results
    
    return results
###################################################################################

def linear_model_predict_and_evaluate (model, set_wrapper, model_type, data_type, results, compute_mean_and_std=True, only_predict=False, only_evaluate=None):
    (x, y) = set_wrapper
    predicted =  model.predict(x) if only_evaluate is None else only_evaluate
    if (only_predict):
        return predicted
    if(data_type == 'A_minus_and_plus'):
        print('A-')
        results = results + evaluation_utilies.slope_test(predicted[:,0], y[:,0], compute_mean_and_std)
        print('A+')
        results = results + evaluation_utilies.slope_test(predicted[:,1], y[:,1], compute_mean_and_std)
    else:
        results = results + evaluation_utilies.slope_test(predicted, y, compute_mean_and_std)
    
    return results
###################################################################################

def train_and_evalute (train_set_wrapper, validation_set_wrapper, test_set_wrapper, validation_seq_wrapper, model_type,
                       data_type, model_id, train_spec, layers_spec,  results_df_path, save_model_path):
    
    model_params, layers_params = NN_model_types.create_model_spec(model_id, train_spec, layers_spec)
    model_id_and_timestamp = model_params['model_id']+'___'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model = NN_model.model_creation_and_fit (train_set_wrapper,  validation_set_wrapper, model_type, data_type, layers_params, model_params, model_id_and_timestamp)

    results = [model_id_and_timestamp, model_type, model_params, layers_spec]
    

    print('\033[1m' + "test evaluation" + '\033[0m')
    results = predict_and_evaluate(model, test_set_wrapper, model_type, data_type, results)
    print('\033[1m' + "validation_seq evaluation" + '\033[0m')
    results = predict_and_evaluate(model, validation_seq_wrapper, model_type, data_type, results)
    
    #save result if requested 
    if(results_df_path is not None):
        locked_file_descriptor = general_utilies.acquireLock()
        results_df = general_utilies.pickle.load(open(results_df_path, 'rb'))
        results_df.loc[len(results_df)] = results
        general_utilies.pickle.dump(results_df, open(results_df_path, 'wb'))
        general_utilies.releaseLock(locked_file_descriptor)
    #save model if requested
    if(save_model_path is not None):
        model.save(save_model_path+model_type+'_'+model_id_and_timestamp+'.h5')
    
    return model
 ################################################################################### 

def train_test_validate_model_type (seq_path, labels_path_minus, labels_path_plus, validate_seq_path, validate_labels_path_minus, validate_labels_path_plus,
                                    model_type, data_type, model_id, train_spec=None, layers_spec=None, results_df_path=None, save_model_path=None, index_for_split=None):

    train_set_wrapper, validation_set_wrapper, test_set_wrapper = NN_load_datasets.load_dataset_model_type (seq_path=seq_path, labels_path_minus=labels_path_minus,
                                                                                                            labels_path_plus= labels_path_plus, model_type=model_type,
                                                                                                            data_type=data_type, split=True,
                                                                                                            index_for_split=index_for_split)

    validation_seq_wrapper = NN_load_datasets.load_dataset_model_type (seq_path=validate_seq_path, labels_path_minus=validate_labels_path_minus,
                                                                       labels_path_plus= validate_labels_path_plus, model_type=model_type,
                                                                       data_type=data_type, split=False)
   
 
    model = train_and_evalute (train_set_wrapper, validation_set_wrapper, test_set_wrapper, validation_seq_wrapper, model_type, data_type, model_id, train_spec, layers_spec,  results_df_path, save_model_path)

    return model, train_set_wrapper, validation_set_wrapper, test_set_wrapper, validation_seq_wrapper
#####################################################################################


#####################################################################################
#############################Evaluation model types##################################
def evaluate_model_type (model_path, seq_path, labels_path_minus, labels_path_plus, validate_seq_path,
                         validate_labels_path_minus, validate_labels_path_plus, model_id, model_type, data_type,
                         compute_mean_and_std=True, secondary_path_minus=None, secondary_path_plus=None,
                         val_secondary_path_minus=None, val_secondary_path_plus=None, seconday_type='all_prob',
                         index_for_split=None, preforme_IG_test=None, preforme_TF_modisco=None, GPU=True):
    if (GPU is not True):
        tf.config.set_visible_devices([], 'GPU')

    #load data and evaluate
    _train_set_wrapper, _validation_set_wrapper, test_set_wrapper = NN_load_datasets.load_dataset_model_type (seq_path=seq_path, labels_path_minus=labels_path_minus, labels_path_plus= labels_path_plus,
                                                                                                              model_type=model_type, data_type=data_type, split=True, secondary_path_minus=secondary_path_minus,
                                                                                                              secondary_path_plus=secondary_path_plus, seconday_type=seconday_type, index_for_split=index_for_split)

    validation_seq_wrapper = NN_load_datasets.load_dataset_model_type (seq_path=validate_seq_path, labels_path_minus=validate_labels_path_minus, labels_path_plus= validate_labels_path_plus,
                                                                       model_type=model_type, data_type=data_type, split=False, secondary_path_minus=val_secondary_path_minus,
                                                                       secondary_path_plus=val_secondary_path_plus, seconday_type=seconday_type)

    #load model
    if type(model_path) is list:
        model_list = [tf.keras.models.load_model(model_path_item, custom_objects={'tf_pearson': NN_utilies.tf_pearson}) for model_path_item in model_path]
    else: 
        model_list = [tf.keras.models.load_model(model_path, custom_objects={'tf_pearson': NN_utilies.tf_pearson})]

    model_num = len(model_list)
    for i in range(model_num):
        predicted = predict_and_evaluate(model_list[i], test_set_wrapper, model_type, data_type, [], compute_mean_and_std, only_predict=True)
        test_predicted = predicted if i==0 else test_predicted + predicted
        predicted = predict_and_evaluate(model_list[i], validation_seq_wrapper, model_type, data_type, [], compute_mean_and_std, only_predict=True)
        validation_predicted = predicted if i==0 else validation_predicted + predicted

    test_predicted = test_predicted/model_num
    validation_predicted = validation_predicted/model_num
    results_dict = {}
    print('\033[1m' + "test evaluation" + '\033[0m')
    results_dict['test'] = predict_and_evaluate(None, test_set_wrapper, model_type, data_type, [], compute_mean_and_std, only_evaluate=test_predicted)
    print(results_dict['test'])
    print('\033[1m' + "validation evaluation" + '\033[0m')
    results_dict['validation_seq'] = predict_and_evaluate(None, validation_seq_wrapper, model_type, data_type, [], compute_mean_and_std, only_evaluate=validation_predicted)
    print(results_dict['validation_seq'])


    if(preforme_IG_test is not None):
        evaluate_IG_test(model_path, validate_seq_path,validate_labels_path_minus,
                         validate_labels_path_plus, model_id, model_type, data_type,
                         secondary_path_minus, secondary_path_plus, preforme_IG_test)
    if (preforme_TF_modisco is not None):
        evaluate_TF_modisco(model_path, seq_path, labels_path_minus, labels_path_plus,
                             model_id,  model_type, data_type, preforme_TF_modisco)

    return results_dict
#####################################################################################

def evaluate_IG_test (model_path, validate_seq_path,validate_labels_path_minus,
                      validate_labels_path_plus, model_id,  model_type, data_type,
                      secondary_path_minus, secondary_path_plus, preforme_IG_test):
    #initilize
    saved_models_path_list = [model_path]
    model_type_list = [model_type]
    model_id_list = [model_id]
    data_type_list = [data_type]
    target_range_list = preforme_IG_test[0] #first item in preforme_IG_test contains the target slice
    pdf_dir = preforme_IG_test[1] #second item in preforme_IG_test contains the folder to save the pdf resutlts 
    output_pdf = False if preforme_IG_test[1] is None else True #output logos into pdf or not
    secondary_list = [preforme_IG_test[2]] if len(preforme_IG_test) == 3 else None #third item (if exist) in preforme_IG_test contains the slice
    #preforme the test
    if(type(target_range_list) is not list):
        target_range_list = [target_range_list]
    for target_range_item in target_range_list:
        NN_utilies.preform_compere_IG_tests (validate_seq_path=validate_seq_path, validate_labels_path_minus=validate_labels_path_minus,
                                             validate_labels_path_plus=validate_labels_path_plus, saved_models_path_list=saved_models_path_list,
                                             model_type_list=model_type_list, model_id_list=model_id_list,data_type_list=data_type_list,
                                             target_range_list=[target_range_item], output_pdf=output_pdf, pdf_dir=pdf_dir,
                                             secondary_list=secondary_list, secondary_path_minus=secondary_path_minus, secondary_path_plus=secondary_path_minus)


def evaluate_TF_modisco (model_path, seq_path, labels_path_minus, labels_path_plus,
                         model_id,  model_type, data_type, preforme_TF_modisco):
    modisco_input_path = general_utilies.files_dir+'modisco_files/input_files/ensemble_' if type(model_path) is list else general_utilies.files_dir+'modisco_files/input_files/'
    target_range = preforme_TF_modisco[0] #first item in preforme_TF_modisco contains the target slice
    batch_size = preforme_TF_modisco[1] # second item in preforme_TF_modisco contains the batch size
    test_validation_train_or_all_set =  preforme_TF_modisco[2] #third is the portion for IG 
    IG_attributions_path = preforme_TF_modisco[3] if len(preforme_TF_modisco) == 4 else None #fourth item (if exist) in preforme_TF_modisco contains the paths for the IG attributions data

    target_range_list = target_range if type(target_range) is list else [target_range]
    for i in range(len(target_range_list)):
        if(IG_attributions_path is None):
            [hyp_impscores, impscores, onehot_data, null_distribution] = TF_modisco.Generate_modisco_Dataset (model_path, seq_path, labels_path_minus, labels_path_plus,
                                                                                            model_id, model_type, data_type,target_range_list[i],
                                                                                            test_validation_train_or_all_set=test_validation_train_or_all_set, save_path=modisco_input_path,
                                                                                            batch_size=batch_size)
        else:
            [hyp_impscores, impscores, onehot_data, null_distribution] = general_utilies.pickle.load(open(IG_attributions_path[i], 'rb'))


        TF_modisco.run_modisco(hyp_impscores, impscores, onehot_data, null_distribution)
        #zip output folder
        modisco_path = general_utilies.files_dir+'modisco_files/'
        zip_name_initial = modisco_path+'output_files/modisco_output_ensemnle_' if type(model_path) is list else modisco_path+'output_files/modisco_output_'
        zipf = TF_modisco.zipfile.ZipFile(zip_name_initial+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_'+model_id+'_'+data_type+'_'+str('all_outputs' if target_range_list[i] is None else target_range_list[i])+'_'+test_validation_train_or_all_set+'_set.zip', 'w', TF_modisco.zipfile.ZIP_DEFLATED)
        TF_modisco.zipdir(modisco_path+'modisco_out/', zipf)
        zipf.close()
        #clean folder
        files = TF_modisco.glob.glob(modisco_path+'modisco_out/*')
        for f in files:
            TF_modisco.os.remove(f)
#####################################################################################


#####################################################################################
#############################Evaluation RESA dataset##################################
def evaluate_RESA_with_linear_model (model_path, data_type, padding=False, aggregation_type="mean"):
    if(type(model_path) is list):
        model_paths = model_path
    else:
        model_paths = [model_path]

    num_of_models = len(model_paths)
    num_of_outputs = 2 if data_type == 'A_minus_and_plus' else 1
    for j in range(num_of_models):
        #load model
        model = tf.keras.models.load_model(model_paths[j], custom_objects={'tf_pearson': NN_utilies.tf_pearson})
        
        if(padding):
            observed_values, features = NN_utilies.load_RESA_data(padding)
            preticted =  model.predict(features)
            predicted_values = preticted if j==0 else predicted_values + preticted
        else:
            observed_values, chunks_num_for_each_seq_df, split_seq_to_chunks_futures_array = NN_utilies.load_RESA_data(padding)
            for i in range(split_seq_to_chunks_futures_array.shape[1]):
                preticted =  model.predict(split_seq_to_chunks_futures_array[:,i,:,:])
                if(i == 0):
                    if(num_of_outputs == 1):
                        preticted_values_matrix = preticted
                    else:
                        preticted_values_matrix = [np.expand_dims(preticted[:,0], axis=1), np.expand_dims(preticted[:,1], axis=1)]
                else:
                    if(num_of_outputs == 1):
                        preticted_values_matrix = np.concatenate((preticted_values_matrix,preticted), axis=1)
                    else:
                        preticted_values_matrix = [np.concatenate((preticted_values_matrix[0], np.expand_dims(preticted[:,0], axis=1)), axis=1), np.concatenate((preticted_values_matrix[1], np.expand_dims(preticted[:,1], axis=1)), axis=1)]
            avg_preticted_values = np.zeros((25043,num_of_outputs))
            for idx, chunks_num in enumerate(chunks_num_for_each_seq_df):
                if(num_of_outputs == 1):
                    if(aggregation_type=='mean'):
                        avg_preticted_values[idx] = sum(preticted_values_matrix[idx, :chunks_num])/chunks_num
                    elif(aggregation_type=='median'):
                        avg_preticted_values[idx] = np.median(preticted_values_matrix[idx, :chunks_num])
                    elif(aggregation_type=='max'):
                        avg_preticted_values[idx] = np.max(preticted_values_matrix[idx, :chunks_num])
                    elif(aggregation_type=='min'):
                        avg_preticted_values[idx] = np.min(preticted_values_matrix[idx, :chunks_num])
                else:
                    if(aggregation_type=='mean'):
                        avg_preticted_values[idx,0] = sum(preticted_values_matrix[0][idx, :chunks_num])/chunks_num
                        avg_preticted_values[idx,1] = sum(preticted_values_matrix[1][idx, :chunks_num])/chunks_num  
                    elif(aggregation_type=='median'):
                        avg_preticted_values[idx,0] = np.median(preticted_values_matrix[0][idx, :chunks_num])
                        avg_preticted_values[idx,1] = np.median(preticted_values_matrix[1][idx, :chunks_num])
                    elif(aggregation_type=='max'):
                        avg_preticted_values[idx,0] = np.max(preticted_values_matrix[0][idx, :chunks_num])
                        avg_preticted_values[idx,1] = np.max(preticted_values_matrix[1][idx, :chunks_num])
                    elif(aggregation_type=='min'):
                        avg_preticted_values[idx,0] = np.min(preticted_values_matrix[0][idx, :chunks_num])
                        avg_preticted_values[idx,1] = np.min(preticted_values_matrix[1][idx, :chunks_num])
                    

            predicted_values = avg_preticted_values if j==0 else predicted_values + avg_preticted_values
        
    predicted_values = predicted_values/num_of_models
    print(predicted_values.shape)
    if(num_of_outputs == 2):
        print("A minus:")
        print("(pearson, p-value):",stats.pearsonr(predicted_values[:,0].reshape((25043, )), observed_values))
        print("A plus:")
        print("(pearson, p-value):",stats.pearsonr(predicted_values[:,1].reshape((25043, )), observed_values))
    else:
        print("(pearson, p-value):",stats.pearsonr(predicted_values.reshape((25043, )), observed_values.reshape((25043, ))))
#####################################################################################