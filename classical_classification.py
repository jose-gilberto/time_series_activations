from aeon.datasets._data_loaders import load_classification
from tsml_eval.estimators.classification.convolution_based import HYDRA, MultiRocketHydra
from aeon.classification.convolution_based._rocket_classifier import RocketClassifier
from tsml_eval.experiments import run_classification_experiment
from tsml_eval.evaluation.storage import load_classifier_results

# from aeon.datasets.tsc_data_lists import univariate_equal_length as dataset_list

# This is the univariate_equal_length of the import above
dataset_list = [
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

# Estimators/Models
estimator = {
    'HYDRA': HYDRA(k=9),
    'Multi-R': RocketClassifier(rocket_transform='MultiRocket'),
    'Hydra-MR': MultiRocketHydra(),
    'RocketClassifier': RocketClassifier(rocket_transform='Rocket'),
}

# Parameters
NUM_EXPERIMENTS = 1

for dataset_name in dataset_list:
    X_train, y_train = load_classification(dataset_name, split='train')
    X_test, y_test = load_classification(dataset_name, split='test')
    # print(dataset_name)

    for current_model in estimator:
        for experiment in range(NUM_EXPERIMENTS):
            # print('****', current_model, '****')
            run_classification_experiment(
                X_train,
                y_train,
                X_test,
                y_test,
                classifier = estimator[current_model],
                results_path = "./experiments/",
                dataset_name=dataset_name,
                resample_id=0,
            )
            
            # # # RESULTS
            
            # cr = load_classifier_results(
            #     "./experiments/HYDRA/Predictions/ACSF1/testResample0.csv"
            # )
            # print(cr.predictions)
            # print(cr.accuracy)
            # print(cr.balanced_accuracy)
            # print(cr.auroc_score)
            # print(cr.log_loss)