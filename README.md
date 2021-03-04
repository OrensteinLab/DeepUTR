# DeepUTR
A new version (a full functioned and full decomentated version) will be uploaded soon.

# Usage
```
usage: DeepUTR.py [-h] [--train TRAIN] [--predict PREDICT]
                  [--evaluate EVALUATE] [--model_type MODEL_TYPE]
                  [--NN_type NN_TYPE] [--data_type DATA_TYPE]
                  [--conventional_model CONVENTIONAL_MODEL]
                  [--input_model_path_1 INPUT_MODEL_PATH_1]
                  [--input_model_path_2 INPUT_MODEL_PATH_2]
                  [--input_sequences INPUT_SEQUENCES]
                  [--input_A_minus_initial INPUT_A_MINUS_INITIAL]
                  [--input_A_plus_initial INPUT_A_PLUS_INITIAL]
                  [--input_A_minus_labels INPUT_A_MINUS_LABELS]
                  [--input_A_plus_labels INPUT_A_PLUS_LABELS]
                  [--input_split_indices INPUT_SPLIT_INDICES]
                  [--output_path OUTPUT_PATH]

DeepUTR - train, predict, or evaluate. Note: folder path must end with '/'

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Perform model type training. Options: '0' (default) -
                        do not perform training. '1' - perform training.
  --predict PREDICT     Perform model type prediction. Avilable only for NN
                        models. Options: '0' (default) - do not perform
                        prediction. '1' - perform prediction.
  --evaluate EVALUATE   Perform model type evaluation. Options: '0' (default)
                        - do not perform evaluation. '1' - perform evaluation.
  --model_type MODEL_TYPE
                        Model_type. Options: 'dynamics' (default) - mRNA
                        degradation dynamics model. 'rate' - mRNA degradation
                        rate model.
  --NN_type NN_TYPE     Neural network architecture. Options: 'CNN' (default)
                        - CNN architecture. 'RNN' - RNN architecture.
  --data_type DATA_TYPE
                        Input 3'UTR sequences data type. Options: 'minus'
                        (default) - non-tailed with poly(A). 'plus' - tailed
                        with poly(A). 'minus_plus' - both non-tailed and
                        tailed with poly(A); for multi-task models.
  --conventional_model CONVENTIONAL_MODEL
                        Conventional model type. Conventional models must be
                        created and trained before used. If a conventional
                        model is used, then 'NN_type' is ignored and
                        'data_type' can not support '-+' input. options:
                        'false' (default) - do not use conventional model.
                        'lasso' - Lasso model. 'RF' - Random Forest model.
  --input_model_path_1 INPUT_MODEL_PATH_1
                        Path of the model. For ensemble model path provide the
                        dirctory path containing the ensemble components (only
                        NN models support ensemble). Default path for NN
                        models are the DeepUTR trained models dirctory
                        corresponding to model_type, NN_type, data_type. For
                        conventional model, you must provide model path, and
                        this is the path of the A- model
  --input_model_path_2 INPUT_MODEL_PATH_2
                        Used only for conventional models. Path of the model
                        A+ model. If needed you must provide this model path
  --input_sequences INPUT_SEQUENCES
                        Path of the sequences input. Default path is the UTR-
                        seq dataset sequences.
  --input_A_minus_initial INPUT_A_MINUS_INITIAL
                        Path of the A- initial mRNA level input for mRNA
                        degradation dynamics models. Used only for predicting.
                        None default path is provided.
  --input_A_plus_initial INPUT_A_PLUS_INITIAL
                        Path of the A+ initial mRNA level input for mRNA
                        degradation dynamics models. Used only for predicting.
                        None default path is provided.
  --input_A_minus_labels INPUT_A_MINUS_LABELS
                        Path of the A- labels input. Used for training and
                        evaluation. Default path is the UTR-seq A- dataset
                        labels.
  --input_A_plus_labels INPUT_A_PLUS_LABELS
                        Path of the A+ labels input. Used for training and
                        evaluation. Default path is the UTR-seq A+ dataset
                        labels.
  --input_split_indices INPUT_SPLIT_INDICES
                        Path of the file containing indices of splitting the
                        dataset for train, validation, and test. Used for
                        training and evaluation. Options: 'false' - do not
                        split the dataset (in this case, in training, the
                        dataset will be splitted randomly). 'random' - random
                        split (see code for detailes). Default option is the
                        path of the file containing indices of splitting the
                        UTR-seq dataset.
  --output_path OUTPUT_PATH
                        Path for the outputs. Default path is the files
                        dirctory of DeepUTR.
```


# Requirements:
The code was tested with:\
Python interpreter == 3.6.6
Python packages required for using DeepUTR:\
   numpy == 1.18.5\
   pandas == 1.1.2\
   scikit-learn == 0.23.0\
   scipy == 1.4.1\
   tensorflow == 2.3.0\

Python packages required for running integrated gradient and TF-MoDISco analysis:\
   modisco==0.5.9.0\
   logomaker==0.8\
   tqdm==4.46.1\
   zipp==3.1.0\

Note: There might be other packages needed. Please contact us in case of any problem.





