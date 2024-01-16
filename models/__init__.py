'''Deep learning algorithms'''

__all__ = [
    "MLPClassifier",
    "MLPRegressor",
    "ResNetClassifier",
    "ResNetRegressor",
    "FCNClassifier",
    "FCNRegressor",
    
]

from .mlp import MLPClassifier, MLPRegressor
from .resnet import ResNetClassifier, ResNetRegressor
from .fcn import FCNClassifier, FCNRegressor
from .inception_time import InceptionTimeClassifier, InceptionTimeRegressor

deeplearning_classifier = {
    'MLPClassifier': MLPClassifier,
    'ResNetClassifier': ResNetClassifier,
    'FCNClassifier': FCNClassifier,
    'InceptionTimeClassifier': InceptionTimeClassifier,
}

deeplearning_regressor = {
    'MLPRegressor': MLPRegressor,
    'ResNetRegressor': ResNetRegressor,
    'FCNRegressor': FCNRegressor,
    'InceptionTimeRegressor': InceptionTimeRegressor,
}