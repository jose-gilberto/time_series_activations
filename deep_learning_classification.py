import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from utils.functions_cls import *
import pandas as pd
from aeon.datasets.tsc_datasets import univariate_equal_length as dataset_list, univariate2015
from aeon.datasets._data_loaders import load_classification, load_from_tsfile
# from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from aeon.datasets.tsc_data_lists import univariate_equal_length as dataset_list
# from pytorch_lightning.loggers.wandb import WandbLogger

# Loading the CUSTOM MODELS into a dict
from models import deeplearning_classifier as custom_estimator

# Experiments and parameters
NUM_EXPERIMENTS = 1
NUM_EPOCHS = 1000
LR = 1e-1
BATCH_SIZE = 16
HIDDEN_CHANNELS = 128
ACTIVATION = nn.ReLU()

# Run the model for these UCR Datasets
datasets = [
    'FordB',
    # 'Symbols',
    # 'CricketZ',
    # 'ChlorineConcentration',
    # 'DistalPhalanxTW',
    # 'Strawberry',
    # 'Worms',
    # 'Wine',
    # 'ProximalPhalanxTW',
    # 'OliveOil',
    # 'ShapeletSim',
    # 'WormsTwoClass',
    # 'ECGFiveDays',
    # 'CinCECGTorso',
    # 'DiatomSizeReduction',
    # 'DistalPhalanxOutlineCorrect',
    # 'ElectricDevices',
    # 'SonyAIBORobotSurface1',
    # 'UWaveGestureLibraryZ',
    # 'Earthquakes',
    # 'ECG200',
    # 'FacesUCR',
    # 'Car',
    # 'ArrowHead',
    # 'Plane',
    # 'ShapesAll',
    # 'Beef',
    # 'ProximalPhalanxOutlineCorrect',
    # 'CBF',
    # 'SwedishLeaf',
    # 'MiddlePhalanxOutlineAgeGroup',
    # 'FaceAll',
    # 'Ham',
    # 'Phoneme',
    # 'HandOutlines',
    # 'NonInvasiveFetalECGThorax1',
    # 'Herring',
    # 'Lightning7',
    # 'ToeSegmentation1',
    # 'UWaveGestureLibraryX',
    # 'DistalPhalanxOutlineAgeGroup',
    # 'StarlightCurves'
]

# Finished Models list
finished_models = [
    # 'MLPClassifier',
    # 'FCNClassifier',
    # 'ResNetClassifier',
    # 'InceptionTimeClassifier',
]


# Logger
# wandb_logger = WandbLogger(log_model="all", project="ActivationFunctions")

results_dict = {
    'dataset': [],
    'model': [],
    'experiment': [],
    'acc': [],
    'f1': []
}

for dataset_name in datasets:
    print('====== DATASET:', dataset_name, "======")

    # ----- AEON DATASETS -----
    X_train, y_train = load_classification(dataset_name, split='train')
    X_test, y_test = load_classification(dataset_name, split='test')
    
    # ----- CUSTOM .TS FILE -----
    # X_train, y_train = load_from_tsfile(full_file_path_and_name='/home/andre/Code/IC/time_series_activations/ts_files/train')
    # X_test, y_test = load_from_tsfile(full_file_path_and_name='/home/andre/Code/IC/time_series_activations/ts_files/test')

    train_label_mapping = {label: idx for idx, label in enumerate(set(y_train))}
    num_classes = len(set(y_train))

    # Lenghts and dimensions
    try:
        sequence_len = X_train.shape[-1]
    except IndexError:
        sequence_len = X_train[0].shape[-1]
    dimension_num = X_train.shape[1]

    # Datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, label_mapping=train_label_mapping)
    test_dataset = TimeSeriesDataset(X_test, y_test, label_mapping=train_label_mapping)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    

    for current_model in custom_estimator:
        if current_model in finished_models: continue

        print('***** MODEL:', current_model, "*****")

        for experiment in range(NUM_EXPERIMENTS):
            
            # Loading Models and Parameters
            model_params = {
                'sequence_len': sequence_len,
                'dimension_num': dimension_num,
                'num_classes': num_classes,
                'out_channels': 128,
                'hidden_channels': HIDDEN_CHANNELS,
                'activation': ACTIVATION,
            }
            # checkpoint_callback = ModelCheckpoint(dirpath='experiments', filename=f"cls_{current_model}_{dataset_name}_{experiment}", verbose=True, monitor='train_loss')
            model = custom_estimator[current_model](**model_params)
            model_classifier = TimeSeriesClassifier(model=model, optimizer=torch.optim.Adadelta(model.parameters(), lr=LR, eps=1e-8))

            # Trainer 
            trainer = Trainer(
                max_epochs=NUM_EPOCHS, 
                accelerator='gpu',
                devices=-1,
                # logger=wandb_logger, 
                # callbacks=[checkpoint_callback],
                # enable_model_summary = False
            )
            
            trainer.fit(model_classifier, train_loader)
            results = trainer.test(model_classifier, test_loader)
            
            results_dict['dataset'].append(dataset_name)
            results_dict['model'].append(current_model)
            results_dict['experiment'].append(experiment)
            results_dict['acc'].append(results[0]['accuracy'])
            results_dict['f1'].append(results[0]['f1'])
            
            results_dataframe = pd.DataFrame(results_dict)
            results_dataframe.to_csv(f'./results_{current_model}_{dataset_name}_exp{experiment:02d}.csv', index=False)
            


            # Finish logging
            # wandb_logger.log_metrics({"model": current_model, "dataset": dataset_name, "experiment": experiment})
            # wandb_logger.finalize("success")

            # Free GPU
            # device = torch.device("cpu")
            # model_classifier.to(device)
            # model = None
            # model_classifier = None
            # torch.cuda.empty_cache()