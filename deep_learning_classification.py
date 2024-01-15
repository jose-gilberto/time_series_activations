import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.functions_cls import *
from aeon.datasets.tsc_data_lists import univariate_equal_length as dataset_list
from aeon.datasets._data_loaders import load_classification
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# CUSTOM MODELS
from models.fcn import FCN
from models.mlp import MLP
from models.resnet import ResNet

# Experiments and parameters
NUM_EXPERIMENTS = 5
NUM_EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 64
ACTIVATION = nn.ReLU()

# Loading the Custom Models into a dict
custom_estimator = {
                    "FCN": FCN,
                    "MLP": MLP,
                    "ResNet": ResNet,
                    }

# UCR Datasets
finished_datasets = [
                        # "ArticularyWordRecognition",
                        # "AtrialFibrillation",
                        # "BasicMotions",
                        # "Cricket",
                        # "DuckDuckGeese",
                        # "EigenWorms",
                        # "Epilepsy",
                        # "EthanolConcentration",
                        # "ERing",
                        # "FaceDetection",
                        # "FingerMovements",
                        # "HandMovementDirection",
                        # "Handwriting",
                        # "Heartbeat",
                        # "Libras",
                        # "LSST",
                        # "MotorImagery",
                        # "NATOPS",
                        # "PenDigits",
                        # "PEMS-SF",
                        # "PhonemeSpectra",
                        # "RacketSports",
                        # "SelfRegulationSCP1",
                        # "SelfRegulationSCP2",
                        # "StandWalkJump",
                        # "UWaveGestureLibrary",
                    ]

finished_models = [
                    # 'CNNClassifier',
                    # 'EncoderClassifier',
                    # 'FCNClassifier',
                    # 'InceptionTimeClassifier',
                    # 'IndividualInceptionClassifier',
                    # 'IndividualLITEClassifier',
                    # 'LITETimeClassifier',
                    # 'MLPClassifier',
                    # 'ResNetClassifier',
                    # 'TapNetClassifier'
                  ]


# Logger
wandb_logger = WandbLogger(log_model="all", project="ActivationFunctions")

for dataset_name in dataset_list:
    if dataset_name in finished_datasets: continue
    # print('====== DATASET:', dataset_name, "======")
    X_train, y_train = load_classification(dataset_name, split='train')
    X_test, y_test = load_classification(dataset_name, split='test')
    train_label_mapping = {label: idx for idx, label in enumerate(set(y_train))}

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
        # print('***** MODEL:', current_model, "*****")
        if current_model in finished_models: continue
        for experiment in range(NUM_EXPERIMENTS):

            # Loading Models and Parameters
            device = torch.device("cuda")
            model_params = {'sequence_len':sequence_len,
                            'dimension_num':dimension_num,
                            'in_channels': dimension_num,
                            'out_channels': BATCH_SIZE,
                            'activation': ACTIVATION}
            checkpoint_callback = ModelCheckpoint(dirpath='experiments', filename=f"{current_model}_{dataset_name}_{experiment}", verbose=True, monitor='val_loss')
            model = custom_estimator[current_model](**model_params).to(device)
            model_classifier = TimeSeriesClassifier(model=model, lr=LR)

            # Trainer 
            trainer = Trainer(max_epochs=NUM_EPOCHS, logger=wandb_logger, callbacks=[checkpoint_callback])
            trainer.fit(model_classifier, train_loader, test_loader)

            # Finish logging
            wandb_logger.log_metrics({"experiment": experiment, "dataset": dataset_name, "model": current_model})
            wandb_logger.finalize("success")

            # Free GPU
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            model_classifier.to(device)
            model = None
            model_classifier = None
            torch.cuda.empty_cache()
