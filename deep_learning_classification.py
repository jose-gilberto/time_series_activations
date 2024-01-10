from utils.functions_cls import *
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from aeon.registry import all_estimators
from aeon.datasets.tsc_data_lists import multivariate_equal_length as multivar_list
from aeon.classification.deep_learning import __all__ as deep_learning_list
from aeon.datasets._data_loaders import load_classification
from torch import nn

from models.fcn import FCN
from models.mlp import MLP
from models.resnet import ResNet


# estimator = all_estimators("classifier", as_dataframe=True)
# estimator = estimator[estimator['name'].isin(deep_learning_list)]

custom_estimator = {
                    "FCN": FCN,
                    # "MLP": MLP,
                    # "ResNet": ResNet,
                    }
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_experiments = 1
num_epochs = 1
lr = 1e-4
batch_size = 64
activation = nn.ReLU()

for dataset_name in multivar_list:
    if dataset_name in finished_datasets: continue
    # print('====== DATASET:', dataset_name, "======")
    X_train, y_train = load_classification(dataset_name, split='train')
    X_test, y_test = load_classification(dataset_name, split='test')
    train_label_mapping = {label: idx for idx, label in enumerate(set(y_train))}

    # Lenghts and dimensions
    sequence_len = len(X_train)
    dimension_num = X_train.shape[1]

    # Datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, label_mapping=train_label_mapping)
    test_dataset = TimeSeriesDataset(X_test, y_test, label_mapping=train_label_mapping)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    for current_model in custom_estimator:
        if current_model in finished_models: continue
        for experiment in range(num_experiments):
            model = custom_estimator[current_model](sequence_len, dimension_num, nn.ReLU()).to(device)
            model_classifier = TimeSeriesClassifier(model=model, lr=lr).to(device)
            trainer = Trainer(max_epochs=num_epochs)
            trainer.fit(model_classifier, train_loader, test_loader)