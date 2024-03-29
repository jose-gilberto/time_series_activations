import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from utils.functions_tser import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from aeon.datasets._data_loaders import load_regression
# from aeon.datasets.tser_data_lists import tser_all as dataset_list
# from pytorch_lightning.loggers.wandb import WandbLogger

# Loading the CUSTOM MODELS into a dict
from models import deeplearning_regressor as custom_estimator

# Experiments and parameters
NUM_EXPERIMENTS = 10
NUM_EPOCHS = 5000
LR = 1e-1
BATCH_SIZE = 16
HIDDEN_CHANNELS = 128
ACTIVATION = nn.ReLU()

# Finished UCR Datasets list
datasets = [
    'AppliancesEnergy', 
    'HouseholdPowerConsumption1',
    'HouseholdPowerConsumption2',
    'BenzeneConcentration',
    'BeijingPM25Quality', 
    'BeijingPM10Quality', 
    'LiveFuelMoistureContent', 
    'FloodModeling1', 
    'FloodModeling2', 
    'FloodModeling3', 
    'AustraliaRainfall', 
    'IEEEPPG', 
    'BIDMC32RR', 
    'BIDMC32HR', 
    'BIDMC32SpO2', 
    'NewsHeadlineSentiment', 
    'NewsTitleSentiment', 
    'Covid3Month',
]


# Finished Models list ( The items in this list WILL NOT BE RUN )
finished_models = [
    "FCNRegressor",
    "MLPRegressor",
    "ResNetRegressor",
    "InceptionTimeRegressor",
]


# Logger
# wandb_logger = WandbLogger(log_model="all", project="ActivationFunctions")

results_dict = {
    'dataset': [],
    'model': [],
    'experiment': [],
    'mse': [],
    'mae': [],
    'rmse': []
}

for dataset_name in datasets:

    X_train, y_train = load_regression(dataset_name, split='train')
    X_test, y_test = load_regression(dataset_name, split='test')

    # Lenghts and dimensions
    try:
        sequence_len = X_train.shape[-1]
    except IndexError:
        sequence_len = X_train[0].shape[-1]
    dimension_num = X_train.shape[1]

    # Datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    for current_model in custom_estimator:
        if current_model in finished_models: continue
        
        for experiment in range(NUM_EXPERIMENTS):

            # Loading Models and Parameters
            device = torch.device("cuda")
            model_params = {
                'sequence_len': sequence_len,
                'dimension_num': dimension_num,
                'out_channels': 128,
                'hidden_channels': HIDDEN_CHANNELS,
                'activation': ACTIVATION,
            }
            # checkpoint_callback = ModelCheckpoint(dirpath='experiments', filename=f"reg_{current_model}_{dataset_name}_{experiment}", verbose=True, monitor='val_loss')
            model = custom_estimator[current_model](**model_params).to(device)
            model_regressor = TimeSeriesRegressor(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=LR, eps=1e-8))

            # Trainer 
            trainer = Trainer(
                max_epochs=NUM_EPOCHS,
                accelerator='gpu',
                devices=-1,
                # logger=wandb_logger,
                # callbacks=[checkpoint_callback], 
                # enable_model_summary = False
            )
            
            trainer.fit(model_regressor, train_loader, test_loader)
            
            results = trainer.test(model_regressor, test_loader)
            
            results_dict['dataset'].append(dataset_name)
            results_dict['model'].append(current_model)
            results_dict['experiment'].append(experiment)
            results_dict['mse'].append(results[0]['mse'])
            results_dict['mae'].append(results[0]['mae'])
            results_dict['rmse'].append(results[0]['rmse'])
            
            results_dataframe = pd.DataFrame(results_dict)
            results_dataframe.to_csv('./ucr_regression.csv', index=False)
            
            
            
            # Finish logging
            # wandb_logger.log_metrics({"model": current_model, "dataset": dataset_name, "experiment": experiment})
            # wandb_logger.finalize("success")

            # Free GPU
            # device = torch.device("cpu")
            # model_regressor.to(device)
            # model = None
            # model_regressor = None
            # torch.cuda.empty_cache()