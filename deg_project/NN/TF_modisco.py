
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, "/home/u30614/deg_project/")

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/data/yaishof/deg_project/")

#######################################################################################
#general imports
import numpy as np
import pandas as pd
from deg_project.general import general_utilies

#imports for run_modisco
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import h5py


#imports for genrating modisco dataset
import tensorflow as tf
from tqdm import tqdm
from deg_project.NN import NN_utilies
from deg_project.NN import NN_load_datasets
from deg_project.NN.NN_IG_imp import get_integrated_gradients
from deg_project.general import modisco_utilies


#define TF-MoDISco resutls path
modisco_path = general_utilies.files_dir+'modisco_files/'

#Save PWM patteren
def savePattern(patten,filename,LEN=70):
    raw_data = {'Pos':np.arange(len(patten))+1,'A': patten[:,0],'C': patten[:,1],'G': patten[:,2],'T': patten[:,3]}
    df = pd.DataFrame(raw_data, columns = ['Pos','A','C','G','T'])
    np.savetxt(filename, df.values, fmt='%i\t%0.6f\t%0.6f\t%0.6f\t%0.6f', delimiter="\t", header="Pos\tA\tC\tG\tT",comments='')


def run_modisco(hyp_impscores, impscores, onehot_data, null_distribution):
    #import TF-MoDISco only here since it's distroying the tf 2 behavior
    import modisco
    import modisco.visualization
    from modisco.visualization import viz_sequence

    #arrange null_distribution as input to the TF-MoDISco if null_distribution exists
    if(null_distribution is None):
        nulldist_args = {}
    else:
        null_distribution = [np.sum(null_distribution_element, axis=1) for null_distribution_element in null_distribution]
        nulldist_perposimp = np.array(null_distribution)
        nulldist_args = {'null_per_pos_scores' : {'task0': nulldist_perposimp}}
    
    #Run TF-MoDISco
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                        #Slight modifications from the default settings
                        target_seqlet_fdr=0.25,
                        seqlets_to_patterns_factory=
                            modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                                kmer_len=6, num_gaps=1,
                                num_mismatches=0,
                                final_min_cluster_size=60
                            )
                        )(
                    #There is only one task, so we just call this 'task0'
                    task_names=["task0"],
                    contrib_scores={'task0': impscores},                
                    hypothetical_contribs={'task0': hyp_impscores},
                    one_hot=onehot_data,
                    **nulldist_args)

    #create Results folder if not exists 
    Path(modisco_path).mkdir(parents=True, exist_ok=True)

    #save the Results
    if(os.path.isfile(modisco_path+"results.hdf5")):
        os.remove(modisco_path+"results.hdf5")

    grp = h5py.File(modisco_path+"results.hdf5")
    tfmodisco_results.save_hdf5(grp)
    hdf5_results = h5py.File(modisco_path+"results.hdf5","r")

    print("Metaclusters heatmap")
    activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
                        np.array(
            [x[0] for x in sorted(
                    enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
                   key=lambda x: x[1])])]
    sns.heatmap(activity_patterns, center=0)
    plt.savefig(modisco_path+"Metaclusters_heatmap.png")

    metacluster_names = [
        x.decode("utf-8") for x in 
        list(hdf5_results["metaclustering_results"]
             ["all_metacluster_names"][:])]

    all_patterns = []
    print(metacluster_names)
    for metacluster_name in metacluster_names:
        print(metacluster_name)
        metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                       [metacluster_name])
        print("activity pattern:",metacluster_grp["activity_pattern"][:])
        all_pattern_names = [x.decode("utf-8") for x in 
                             list(metacluster_grp["seqlets_to_patterns_result"]
                                                 ["patterns"]["all_pattern_names"][:])]
        if (len(all_pattern_names)==0):
            print("No motifs found for this activity pattern")
        for i,pattern_name in enumerate(all_pattern_names):
            print(metacluster_name, pattern_name)
            all_patterns.append((metacluster_name, pattern_name))
            pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
            print("total seqlets:",len(pattern["seqlets_and_alnmts"]["seqlets"]))
            background = np.array([0.27, 0.23, 0.23, 0.27])
            print("Hypothetical scores:")
            viz_sequence.plot_weights(pattern["task0_hypothetical_contribs"]["fwd"])
            plt.savefig(modisco_path+'modisco_out/'+'Hypothetical'+str(i)+'.png')
            print("Actual importance scores:")
            viz_sequence.plot_weights(pattern["task0_contrib_scores"]["fwd"])
            plt.savefig(modisco_path+'modisco_out/'+'importance'+str(i)+'.png')
            print("onehot, fwd and rev:")
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                            background=background)) 
            plt.show()
            plt.savefig(modisco_path+'modisco_out/'+'onehot_fwd'+str(i)+'.png')
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
                                                            background=background))
            plt.savefig(modisco_path+'modisco_out/'+'onehot_rev'+str(i)+'.png')
            savePattern(np.array(pattern["task0_hypothetical_contribs"]["fwd"]),modisco_path+"modisco_out/hyp_pattern"+str(i)+".txt")
            savePattern(np.array(pattern["task0_contrib_scores"]["fwd"]),modisco_path+"modisco_out/imp_pattern"+str(i)+".txt")
            savePattern(np.array(pattern["sequence"]["fwd"]),modisco_path+"modisco_out/onehot_pattern"+str(i)+".txt")

    hdf5_results.close()


#######################################################################################
#generate dataset importance scores for modisco
def generate_dataset_importance_scores(model, sequences_fetures, intial_features, target_range, batch_size=25, create_null_distribution=True):
    #compute both importance score (ex) and hypothetical importance score (hyp_ex)
    ex_list, hyp_ex_list = compute_impratnce_scores(model, sequences_fetures, intial_features, target_range, batch_size)

    #create null distribution 
    null_distribution = create_null_distribution (model, sequences_fetures, intial_features, target_range, batch_size)

    return sequences_fetures, ex_list, hyp_ex_list, null_distribution

#compute both importance score and hypothetical importance score
def compute_impratnce_scores(model, sequences_fetures, intial_features, target_range, batch_size):
    ex_list = []
    hyp_ex_list = []
    part_nums = (len(sequences_fetures)//batch_size)+1 #devide the calculation acordding to the bach size 
    for sequences_fetures_part, intial_features_part in tqdm(zip(np.array_split(sequences_fetures, part_nums), np.array_split(intial_features, part_nums))):
        for i  in range(len(model)):
            ex_to_add, hyp_ex_to_add = get_integrated_gradients(model[i], sample_inputs=[sequences_fetures_part, intial_features_part], target_range=target_range, multiple_samples=True)
            ex_to_add, hyp_ex_to_add = ex_to_add[0], hyp_ex_to_add[0] #[0] for taking only sequence importance
            ex = ex_to_add if i == 0 else ex + ex_to_add
            hyp_ex = hyp_ex_to_add if i == 0 else hyp_ex + hyp_ex_to_add

        #in case of ensemble model, take the mean of IG scores
        ex, hyp_ex = ex/len(model), hyp_ex/len(model) #take the mean
        ex_list = ex_list + list(ex)
        hyp_ex_list = hyp_ex_list + list(hyp_ex)

    return ex_list, hyp_ex_list

#create null distribution for modisco
def create_null_distribution (model, sequences_fetures, intial_features, target_range, batch_size): 
    if (create_null_distribution):
        null_distribution = []
        sequences_fetures_for_dinuc_shuffle, _,  intial_features_dinuc_shuffle, _ = train_test_split(sequences_fetures, intial_features, test_size=0.5, random_state=42)
        for i in range(len(sequences_fetures_for_dinuc_shuffle)):
            sequences_fetures_for_dinuc_shuffle [i] = modisco_utilies.dinuc_shuffle(sequences_fetures_for_dinuc_shuffle[i], seed=42)
        
        part_nums = (len(sequences_fetures_for_dinuc_shuffle)//batch_size)+1
        for sequences_fetures_for_dinuc_shuffle_part, intial_features_dinuc_shuffle_part in tqdm(zip(np.array_split(sequences_fetures_for_dinuc_shuffle, part_nums), np.array_split(intial_features_dinuc_shuffle, part_nums))):
            for i  in range(len(model)):
                ex_to_add, _hyp_ex_to_add = get_integrated_gradients(model[i], sample_inputs=[sequences_fetures_for_dinuc_shuffle_part, intial_features_dinuc_shuffle_part], target_range=target_range, multiple_samples=True)
                ex_to_add = ex_to_add[0] #[0] for taking only sequence importance
                ex = ex_to_add if i == 0 else ex + ex_to_add

            ex = ex/len(model) #take the mean
            null_distribution = null_distribution + list(ex)
    else:
        null_distribution = None
    
    return null_distribution


#generate modisco Dataset
def generate_modisco_dataset(model_path, seq_path, labels_path_minus, labels_path_plus, model_id, model_type,
                              data_type, target_range=None, test_validation_train_or_all_set='test',
                              save_path=None, index_for_split=None, batch_size=256):
    #load model or list of models 
    if type(model_path) is list:
        model = [tf.keras.models.load_model(model_path_item, custom_objects={'tf_pearson': NN_utilies.tf_pearson}) for model_path_item in model_path]
    else: 
        model = [tf.keras.models.load_model(model_path, custom_objects={'tf_pearson': NN_utilies.tf_pearson})]

    #load the requested dataset 
    split = False if test_validation_train_or_all_set=='all' else True
    data_sets = NN_load_datasets.load_dataset_model_type (seq_path=seq_path, labels_path_minus=labels_path_minus,
                                                          labels_path_plus=labels_path_plus,
                                                          model_type=model_type, data_type=data_type, split=split, index_for_split=index_for_split)
    #unpack the dataset                                               
    if (test_validation_train_or_all_set == 'all'):
        (initial_values_features, one_hot_features, _) = data_sets
    else:
        train_set_wrapper, validation_set_wrapper, test_set_wrapper = data_sets
        if(test_validation_train_or_all_set == 'test'):
            (initial_values_features, one_hot_features, _) = test_set_wrapper
        elif(test_validation_train_or_all_set == 'validation'):
            (initial_values_features, one_hot_features, _) = validation_set_wrapper
        elif(test_validation_train_or_all_set == 'train'):
            (initial_values_features, one_hot_features, _) = train_set_wrapper
        elif(test_validation_train_or_all_set == 'validation_and_test'):
            (initial_values_features_v, one_hot_features_v, _) = validation_set_wrapper
            (initial_values_features_t, one_hot_features_t, _) = test_set_wrapper
            initial_values_features, one_hot_features = np.concatenate((initial_values_features_v, initial_values_features_t)), np.concatenate((one_hot_features_v, one_hot_features_t))
        else:
            raise ValueError('invalid test_validation_train_or_all_set')

    onehot_data, impscores, hyp_impscores, null_distribution = generate_dataset_importance_scores(model, one_hot_features, initial_values_features, target_range, batch_size)

    #save the resutls in neeeded
    if (save_path is not None):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        general_utilies.pickle.dump([hyp_impscores, impscores, onehot_data, null_distribution], open(save_path+'modisco_input_'+model_id+'_'+data_type+'_'+str('all_outputs' if target_range is None else target_range)+'_'+test_validation_train_or_all_set+'_set.sav', 'wb'))
    
    return [hyp_impscores, impscores, onehot_data, null_distribution]
