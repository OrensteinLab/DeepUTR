
#general imports
import pandas as pd
import numpy as np
import re
import math
import statistics
import warnings
from deg_project.general import general_utilies
from deg_project.general import sequence_utilies
import tensorflow as tf

#imports for IG
from deg_project.NN.NN_IG_imp import get_integrated_gradients
from deg_project.NN import NN_load_datasets
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


###########################################################################
#########################general utilies####################################
# pearson correlation metric
def tf_pearson(x, y):    
    mx = tf.math.reduce_mean(input_tensor=x)
    my = tf.math.reduce_mean(input_tensor=y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(input_tensor=tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return  r_num / r_den
############################################################################


############################################################################
#########################utilies for IG#####################################
#IntegratedGradients
def run_IG (model, one_hot_features, intial_features=None, target_range=None, output_pdf_name=None, multiple_samples=True):
    #define parameters for the process
    Logo_letters = ['A', 'C', 'G', "T"]
    secondary_color = False
    starting_index = 0
    ending_index = 4

    #if intial_features=None then compute IG only on the sequence features 
    if (intial_features is not None):
        if type(model) is list:
            explained_seq_fetures_list = [get_integrated_gradients(model_i, sample_inputs=[one_hot_features, intial_features], target_range=target_range, multiple_samples=multiple_samples)[0][0] for model_i in model] # [0][0] means that it takes only IG importance (first [0]) for the seqence (second [0])
            explained_seq_fetures =sum(explained_seq_fetures_list)/len(model) #take the average
        else:
            explained_seq_fetures, _ = get_integrated_gradients(model, sample_inputs=[one_hot_features, intial_features], target_range=target_range, multiple_samples=multiple_samples)
            explained_seq_fetures = explained_seq_fetures[0] #take only the IG importance
    else:
        if type(model) is list:
            explained_seq_fetures_list = [get_integrated_gradients(model_i, sample_inputs=one_hot_features, target_range=target_range, multiple_samples=multiple_samples)[0] for model_i in model] # [0] means that it takes only IG importance
            explained_seq_fetures =sum(explained_seq_fetures_list)/len(model) #take the average 
        else:
            explained_seq_fetures, _ = get_integrated_gradients(model, sample_inputs=one_hot_features, target_range=target_range, multiple_samples=multiple_samples)

    explained_seq_fetures_letters = explained_seq_fetures[:, :, starting_index:ending_index] if multiple_samples else explained_seq_fetures[:, starting_index:ending_index]

    #save the resutls in a pdf file if needed
    if(output_pdf_name is not None):
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_pdf_name)
        explained_seq_fetures_letters_list = explained_seq_fetures_letters if multiple_samples else [explained_seq_fetures_letters] #
        for explained_seq_fetures_letters_item in explained_seq_fetures_letters_list:
            # create Logo object
            explained_seq_fetures_letters_item = pd.DataFrame(explained_seq_fetures_letters_item, columns=Logo_letters)
            IG_logo = sequence_utilies.create_DNA_logo (PWM_df=explained_seq_fetures_letters_item, secondary_color=secondary_color)
            pdf.savefig(IG_logo.ax.figure)
            plt.close('all')
        pdf.close()

    return explained_seq_fetures_letters
############################################################################

def compute_IG_PWM (model, validate_seq_path, validate_labels_path_minus, validate_labels_path_plus, model_type, model_id, data_type, target_range=None,
                    output_pdf=False, pdf_dir=''):
    output_pdf_name = pdf_dir+model_id+'_'+data_type+'_gradient_'+str('all_outputs' if target_range is None else target_range)+'.pdf' if output_pdf else None 

    validation_set = NN_load_datasets.load_dataset_model_type (seq_path=validate_seq_path, labels_path_minus=validate_labels_path_minus,
                                                               labels_path_plus=validate_labels_path_plus, model_type=model_type,
                                                               data_type=data_type, split=False)
    if(model_type == 'dynamics'):
        (initial_values_features, one_hot_features, _) = validation_set
    else:
        (one_hot_features, _) = validation_set
        initial_values_features = None

    explained_seq_fetures_letters_list = run_IG(model, one_hot_features, initial_values_features, target_range, output_pdf_name)

    return explained_seq_fetures_letters_list
###########################################################################

def create_ideal_IG (validation_sequences, test_type):
    num_of_samples = len(validation_sequences)
    #shortest overlap
    seqence_elements = ['ATAGTTATTTATTTATTTTAAC', #ARE destabilizing
                        #'ACTTTCCCAACGATCAAAATAAC', #bg ?
                        'GAAGCACTTCAT', #m430 destabilizing
                        #'ATCAGAAAAAAAAAAAAAGAGTC', #polyA ?
                        'TTCTTTCTTTTTTTTTTTTTGCCTG', #polyU stabilizing 
                        'TCATATGTAAATATGTACATATTAC'] #PUM destabilizing
    elemet_factors = [-1,-1,1,-1] #stabilize (1) or destabilizing (-1) 
    
    seqence_elements_one_hot= list(map(sequence_utilies.one_hot_encoding, seqence_elements))
    # Create a zip object from two lists
    zipbObj = zip(seqence_elements, seqence_elements_one_hot)
    # Create a dictionary from zip object
    seqence_elements_to_one_hot_dict = dict(zipbObj)
        
    ideal_IG_list = []
    for index in range(num_of_samples):
        seq = validation_sequences[index]
        if (test_type == 'pearson' or test_type == 'L2'):
            ideal_IG = np.zeros((len(seq),4)) #initialize first with 0
        else:
            raise ValueError('invalid test type')
        
        for seqence_element, elemet_factor in zip(seqence_elements, elemet_factors):
            seqence_element_length = len (seqence_element)
            starting_indexes = [m.start() for m in re.finditer('(?='+seqence_element+')', seq)]
            for starting_index in starting_indexes:
                ideal_IG[starting_index:starting_index+seqence_element_length,:] = elemet_factor*seqence_elements_to_one_hot_dict[seqence_element]
        ideal_IG = np.sum(ideal_IG, axis=1)
        ideal_IG_list.append(ideal_IG)
    
    return ideal_IG_list
###########################################################################

def compute_test_value_between_ideal_IG_and_actucal_IG(actual_IG, ideal_IG, test_type):
    actual_IG = np.sum(actual_IG, axis=1)
    actual_IG = np.squeeze(np.asarray(actual_IG))
    ideal_IG = np.squeeze(np.asarray(ideal_IG))
    
    result = None
    if(test_type == 'pearson'):
        #No need to normalize as the metric is pearson 
        result = general_utilies.pearsonr(actual_IG, ideal_IG)[0]
    elif(test_type == 'L2'):
        normalized_actual_IG = actual_IG/np.amax(np.abs(actual_IG)) #normalize by absulte max value
        result = np.linalg.norm(ideal_IG-normalized_actual_IG)
    else:
        raise ValueError('invalid test type')

    return result
##########################################################################

def IG_test (test_type, tested_models_actucal_IG_list, model_id_list, data_type_list, target_range_list, print_only_mean_results):
    num_of_models = len(tested_models_actucal_IG_list)
    validation_sequences = pd.read_csv(general_utilies.validation_seq_PATH)['seq']
    ideal_IG_list = create_ideal_IG (validation_sequences, test_type)

    counters = [0] * num_of_models
    result_values_list = [[] for _ in range(num_of_models)] #contain all result values for mean and std compute
    sequences_to_exclude_from_counting = [3,11,19,28,36,44,53] + [9,12,16,17,20,21,34,37,38] #[contain unkown motif] + [only bg or polyA]
    
    for i in range(len(ideal_IG_list)):
        result_values = [compute_test_value_between_ideal_IG_and_actucal_IG(tested_model_actucal_IG_list[i], ideal_IG_list[i], test_type) for tested_model_actucal_IG_list in tested_models_actucal_IG_list]
        for j in range(num_of_models):
            result_values_list[j].append(result_values[j])
        if(test_type == 'pearson'):
            value = max(result_values)
        else:
            value = min(result_values)
        index = result_values.index(value) 

        if ( not (i in sequences_to_exclude_from_counting)):
            if (math.isnan(value)):
                warnings.warn("Warning: resulted with nan")
            else:
                counters[index] = counters[index] + 1
        
        if(print_only_mean_results == False):    
            print("id ", i,": ", index, value)
    
    target_range_list = ['all_outputs' if target_range is None else target_range for  target_range in target_range_list]
    output_names = [model_id_list[i]+'_'+data_type_list[i]+'_gradient_'+str(target_range_list[i]) for i in range(num_of_models)]

    test_metric_results = []
    for i in range(num_of_models):
        result_values = [x for x in result_values_list[i] if str(x) != 'nan']
        test_metric_results.append(statistics.mean(result_values))
        print(output_names[i],':', counters[i], "(mean:", statistics.mean(result_values)," std:", statistics.stdev(result_values),")")

    return test_metric_results
###########################################################################

def preform_compere_IG_tests (validate_seq_path, validate_labels_path_minus, validate_labels_path_plus, saved_models_path_list,
                              model_type_list, model_id_list, data_type_list, target_range_list, output_pdf, pdf_dir, print_only_mean_results=True,
                              metric_types=["pearson"]):
    tested_models_actucal_IG_list = preform_IG_on_model_list (validate_seq_path, validate_labels_path_minus, validate_labels_path_plus, saved_models_path_list,
                                                              model_type_list, model_id_list, data_type_list, target_range_list, output_pdf, pdf_dir)
    test_metrics_results = []
    for metric_type in metric_types:
        test_metric_results = IG_test (metric_type, tested_models_actucal_IG_list,model_id_list, data_type_list, target_range_list, print_only_mean_results)
        test_metrics_results.append (test_metric_results)
    
    if len(test_metrics_results)==1:
        return test_metrics_results[0]
    else:
        return test_metrics_results
############################################################################

def preform_IG_on_model_list (validate_seq_path, validate_labels_path_minus, validate_labels_path_plus, saved_models_path_list,
                              model_type_list, model_id_list, data_type_list, target_range_list, output_pdf, pdf_dir):
    tested_models_actucal_IG_list = []
    for i in range(len(model_type_list)):
        if type(saved_models_path_list[i]) is list:
            model = [tf.keras.models.load_model(saved_models_path, custom_objects={'tf_pearson': tf_pearson}) for saved_models_path in  saved_models_path_list[i]]
        else:
            model = tf.keras.models.load_model(saved_models_path_list[i], custom_objects={'tf_pearson': tf_pearson})
        tested_models_actucal_IG_list.append(compute_IG_PWM (model=model, validate_seq_path=validate_seq_path, validate_labels_path_minus=validate_labels_path_minus,
                                                            validate_labels_path_plus=validate_labels_path_plus, model_type=model_type_list[i], model_id=model_id_list[i],
                                                            data_type=data_type_list[i],target_range=target_range_list[i], output_pdf=output_pdf, pdf_dir=pdf_dir))
    
    return tested_models_actucal_IG_list
############################################################################


############################################################################
#########################utilies for RESA Test##############################
def one_hot_encoding_of_seq_list(list_of_seq):
    return list(map(sequence_utilies.one_hot_encoding, list_of_seq)) 
############################################################################

#returns list which contains the string in chunks of length "length"
def chunkstring(string, length):
    return list((string[0+i:length+i] for i in range(0, len(string), length)))
############################################################################

def pad_chunks (seq_chunks, size_of_chucnk, max_num_of_chunks):
    all_cheunks_feature_matrix = np.zeros((max_num_of_chunks, size_of_chucnk, 4)).astype('uint8')
    for idx, chunk_feature_marix in enumerate(seq_chunks):
        all_cheunks_feature_matrix[idx, :chunk_feature_marix.shape[0], :chunk_feature_marix.shape[1]] = chunk_feature_marix
    
    return all_cheunks_feature_matrix
############################################################################

def split_seq_to_chunks (seq_df, size_of_chucnk):
    max_seq_len = seq_df.map(len).max()
    max_num_of_chunks = math.ceil(max_seq_len/size_of_chucnk)
    seq_df_splited = seq_df.apply(chunkstring, args=(size_of_chucnk,))
    seq_df_splited_one_hot = seq_df_splited.apply(one_hot_encoding_of_seq_list)
    seq_df_splited_one_hot= seq_df_splited_one_hot.apply(np.asarray) #convert from list of one hot encoded matrix to array of one hot encoded matrix
    seq_df_splited_one_hot_padded = seq_df_splited_one_hot.apply(pad_chunks, args=(size_of_chucnk, max_num_of_chunks, ) ) #zero pading of each element to shape (max_num_of_chunks, size_of_chucnk, 4)
    split_seq_to_chunks_futures_array = np.array(seq_df_splited_one_hot_padded.tolist()) #convert to array with shape (df_size, max_num_of_chunks, size_of_chucnk, 4)
    
    chunks_num_for_each_seq_df = (seq_df.map(len)/size_of_chucnk).map(math.ceil) #calcualte for each seq the number of chunks needed
    
    return chunks_num_for_each_seq_df, split_seq_to_chunks_futures_array
############################################################################

def load_RESA_data (padding=False):
    #load genome rec data
    df = pd.read_csv (general_utilies.RESA_data_PATH, sep="\t")
    df = df.drop(6377) #sequence that contains Ns
    seq_df = df ["WindowSeq"]
    observed_values = np.array(df ["WindowRESA"])

    if(padding):
        seq = seq_df.values.tolist()
        features_list = [sequence_utilies.one_hot_encoding(x) for x in seq]
        features = tf.keras.preprocessing.sequence.pad_sequences(features_list, maxlen=901, padding='post', dtype="uint8")  #901 is the max len of RESA seqence.
        return observed_values, features

    #create features for the model
    chunks_num_for_each_seq_df, split_seq_to_chunks_futures_array = split_seq_to_chunks(seq_df, 110)

    return observed_values, chunks_num_for_each_seq_df, split_seq_to_chunks_futures_array
###########################################################################