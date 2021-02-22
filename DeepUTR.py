import argparse
import glob
from deg_project.general import general_utilies
from deg_project.NN import NN_train_test_models_utilies
from deg_project.lasso_RF import RF_and_lasso_train_test_models_utilies

def parser_func():
    parser = argparse.ArgumentParser(description="DeepUTR - train, predict, or evaluate. Note: folder path must add with '/'")
    parser.add_argument('--train', type=int, default=0,
                        help="Perform model type training. Options: '0' (default) - do not perform training. '1' - perform training.")
    parser.add_argument('--predict', type=int, default=0,
                        help="Perform model type prediction. Avilable only for NN models. Options: '0' (default) - do not perform prediction. '1' - perform prediction.")
    parser.add_argument('--evaluate', type=int, default=0,
                        help="Perform model type evaluation. Options: '0' (default) - do not perform evaluation. '1' - perform evaluation.")

    parser.add_argument('--model_type', type=str, default='dynamics',
                        help="Model_type. Options: 'dynamics' (default) - mRNA degradation dynamics model. 'rate' - mRNA degradation rate model.")
    parser.add_argument('--NN_type', type=str, default='CNN',
                        help="Neural network architecture. Options: 'CNN' (default) - CNN architecture. 'RNN' - RNN architecture.")
    parser.add_argument('--data_type', type=str, default="minus",
                        help="Input 3'UTR sequences data type. Options: 'ninus' (default) - non-tailed with poly(A). 'plus' - tailed with poly(A). 'minus_plus' - both non-tailed and tailed with poly(A); for multi-task models.")
    parser.add_argument('--conventional_model', type=str, default='false',
                        help="Conventional model type. Conventional models must be created and trained before used. If a conventional model is used, then 'NN_type' is ignored and 'data_type' can not support '-+' input. options: 'false' (default) - do not use conventional model. 'lasso' - Lasso model. 'RF' - Random Forest model.")

    parser.add_argument('--input_model_path_1', type=str, default="default",
                        help="Path of the model. For ensemble model path provide the dirctory path containing the ensemble components (only NN models support ensemble). Default path for NN models are the DeepUTR trained models dirctory corresponding to model_type, NN_type, data_type. For conventional model, you must provide model path, and this is the path of the A- model")
    parser.add_argument('--input_model_path_2', type=str, default="false",
                        help="Used only for conventional models. Path of the model A+ model. If needed you must provide this model path")

    parser.add_argument('--input_sequences', type=str, default=general_utilies.seq_PATH,
                        help="Path of the sequences input. Default path is the UTR-seq dataset sequences.")

    parser.add_argument('--input_A_minus_initial', type=str, default='false',
                        help="Path of the A- initial mRNA level input for mRNA degradation dynamics models. Used only for predicting. None default path is provided.")
    parser.add_argument('--input_A_plus_initial', type=str, default='false',
                        help="Path of the A+ initial mRNA level input for mRNA degradation dynamics models. Used only for predicting. None default path is provided.")

    parser.add_argument('--input_A_minus_labels', type=str, default=general_utilies.A_minus_normalized_levels_PATH,
                        help="Path of the A- labels input. Used for training and evaluation. Default path is the UTR-seq A- dataset labels.")
    parser.add_argument('--input_A_plus_labels', type=str, default=general_utilies.A_plus_normalized_levels_PATH,
                        help="Path of the A+ labels input. Used for training and evaluation. Default path is the UTR-seq A+ dataset labels.")

    parser.add_argument('--input_split_indices', type=str, default=general_utilies.split_to_train_validation_test_disjoint_sets_ids_PATH,
                        help="Path of the file containing indices of splitting the dataset for train, validation, and test. Used for training and evaluation. Options: 'false' - do not split the dataset (in this case, in training, the dataset will be splitted randomly). 'random' - random split (see code for detailes). Default option is the path of the file containing indices of splitting the UTR-seq dataset.")

    parser.add_argument('--output_path', type=str, default=general_utilies.files_dir,
                        help="Path for the outputs. Default path is the files dirctory of DeepUTR.")
    args = parser.parse_args()

    return args

def train(args, data_type):
    index_for_split = None if (args.input_split_indices=='false' or args.input_split_indices=='random') else args.input_split_indices # Random split if args.input_split_indices is 'false' or 'random
    
    if (args.conventional_model == 'false'):
        model_id = dynamics_model_id_decoder(args)
        NN_train_test_models_utilies.train_test_validate_model_type (
            seq_path=args.input_sequences,
            labels_path_minus=args.input_A_minus_labels,
            labels_path_plus=args.input_A_plus_labels,
            model_type=args.model_type,
            data_type=data_type,
            model_id=model_id,
            save_model_path=args.output_path,
            index_for_split=index_for_split
        )
    else:
        labels_path_minus, labels_path_plus = rate_model_labels_decoder(args, data_type)
        RF_and_lasso_train_test_models_utilies.train_test_validate_lasso_or_RF_model(
            seq_path=args.input_sequences,
            labels_path_minus=labels_path_minus,
            labels_path_plus=labels_path_plus,
            model_type=args.model_type,
            lasso_or_RF=args.conventional_model,
            index_for_split=index_for_split,
            save_model_path=args.output_path
        )

def evaluate(args, data_type):
    if args.input_split_indices != 'false':
        split = True
        index_for_split = None if args.input_split_indices=='random' else args.input_split_indices # Random split if args.input_split_indices=='random'
    else:
        split = False
        index_for_split = None

    if (args.conventional_model == 'false'):
        model_id = dynamics_model_id_decoder(args)
        model_path = dynamics_model_path_decoder(args, data_type)
        NN_train_test_models_utilies.evaluate_model_type (
            model_path=model_path,
            seq_path=args.input_sequences,
            labels_path_minus=args.input_A_minus_labels,
            labels_path_plus=args.input_A_plus_labels,
            model_id=model_id,
            model_type=args.model_type,
            data_type=data_type,
            index_for_split=index_for_split,
            split=split
            )
    else:
        labels_path_minus, labels_path_plus = rate_model_labels_decoder(args, data_type)
        model_A_minus_path, model_A_plus_path = rate_model_path_decoder(args, data_type)
        RF_and_lasso_train_test_models_utilies.evaluate_lasso_or_RF_model (
            model_A_minus_path=model_A_minus_path,
            model_A_plus_path=model_A_plus_path,
            seq_path=args.input_sequences,
            labels_path_minus=labels_path_minus,
            labels_path_plus=labels_path_plus,
            model_type=args.model_type,
            lasso_or_RF=args.conventional_model,
            split=split,
            index_for_split=index_for_split)
            
def predict(args, data_type):
    if (args.conventional_model == 'false'):
        model_id = dynamics_model_id_decoder(args)
        model_path = dynamics_model_path_decoder(args, data_type)
        NN_train_test_models_utilies.evaluate_model_type (
            model_path=model_path,
            seq_path=args.input_sequences,
            labels_path_minus=args.input_A_minus_labels,
            labels_path_plus=args.input_A_plus_labels,
            model_id=model_id,
            model_type=args.model_type,
            data_type=data_type,
            only_predict=True,
            split=False,
            output_path=args.output_path
            )
    else:
        print ('We do not suporrt prediction option for the conventional models')

def dynamics_model_id_decoder(args):
    if (args.model_type == 'dynamics'):
        if (args.NN_type == 'CNN'):
            model_id = 'model_8_points_id_1'
        elif(args.NN_type == 'RNN'):
            model_id = 'model_8_points_id_2'
        else:
            raise ValueError('invalid NN_type')
    elif (args.model_type == 'rate'):
        if (args.NN_type == 'CNN'):
            model_id = 'linear_model_points_id_1'
        elif(args.NN_type == 'RNN'):
            model_id = 'linear_model_points_id_2'
        else:
            raise ValueError('invalid NN_type')
    else:
        raise ValueError('invalid model_type')

    return model_id

def dynamics_model_path_decoder(args, data_type):
    model_path = args.input_model_path_1
    if (model_path == "default"):
        if (args.model_type == 'dynamics'):
            model_path = general_utilies.files_dir+'saved_models_8_disjoint/ensemble/'
        else:
            model_path = general_utilies.files_dir+'saved_models_linear_disjoint/ensemble/'
        model_path = model_path + args.model_type+'_'+data_type+'_'+args.NN_type+'/'
    if (model_path[-1] == '/'):
        model_path = glob.glob(model_path+"*") # list of all fiels in the directory
    
    return model_path

def rate_model_labels_decoder(args, data_type):
    if (data_type == '-' or data_type == '-+'):
        labels_path_minus = args.input_A_minus_labels
    else:
        labels_path_minus = None
    if (data_type == '+' or data_type == '-+'):
        labels_path_plus = args.input_A_plus_labels
    else:
        labels_path_plus = None

    return labels_path_minus, labels_path_plus

def rate_model_path_decoder(args, data_type):
    if (data_type == '-' or data_type == '-+'):
        model_A_minus_path = args.input_model_path_1
    else:
        model_A_minus_path = None
    if (data_type == '+' or data_type == '-+'):
        model_A_plus_path = args.input_model_path_2
    else:
        model_A_plus_path = None

    return model_A_minus_path, model_A_plus_path

def data_type_decoder(args):
    if(args.data_type == 'minus'):
        data_type = '-'
    elif(args.data_type == 'plus'):
        data_type = '+'
    elif(args.data_type == 'minus_plus'):
        data_type = data_type = '-+'
    else:
        raise ValueError('invalid data_type')
    
    return data_type
    
def main():
    args = parser_func()
    data_type = data_type_decoder(args)
    if (args.train == 1):
        print("performing train")
        train(args, data_type)
    elif (args.evaluate == 1):
        print("performing evaluation")
        evaluate(args, data_type)
    elif(args.predict == 1):
        print("performing prediction")
        predict(args, data_type)
    else:
        raise ValueError('None of the arguments train, evaluate, and predict got Valid input') 


if __name__ == "__main__":
    main()



