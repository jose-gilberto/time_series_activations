from aeon.datasets._data_loaders import load_regression
from tsml_eval.experiments import run_regression_experiment
from tsml_eval.evaluation.storage import load_regressor_results
from tsml_eval.estimators.regression.convolution_based import MultiRocketHydra
# from tsml_eval._wip.hc2_regression.hivecote_v2 import HIVECOTEV2
# from aeon.regression.deep_learning import InceptionTimeRegressor
from tsml.shapelet_based import RDSTRegressor
# from  import WEASEL

# from aeon.datasets.tsc_data_lists import tser_soton as dataset_list

# This is the tser_soton of the import above
dataset_list = [
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

# Estimators/Models
estimator = {
    # 'HC2': HIVECOTEV2(),
    'Hydra-MR': MultiRocketHydra(),
    # 'InceptionT': InceptionTimeRegressor(),
    'RDST': RDSTRegressor(),
    # 'WEASEL': WEASEL_V2(),
}

# Parameters
NUM_EXPERIMENTS = 1

for dataset_name in dataset_list:
    X_train, y_train = load_regression(dataset_name, split='train')
    X_test, y_test = load_regression(dataset_name, split='test')
    # print(dataset_name)

    for current_model in estimator:
        for experiment in range(NUM_EXPERIMENTS):
            print('****', current_model, '****')
            run_regression_experiment(
                X_train,
                y_train,
                X_test,
                y_test,
                regressor = estimator[current_model],
                results_path = "./experiments/",
                dataset_name=dataset_name,
                resample_id=0,
            )
            
            # # # RESULTS
            
            # cr = load_regressor_results(
            #     f"./experiments/{current_model}/Predictions/{dataset_name}/testResample0.csv"
            # )
            # print(cr.predictions)
            # print(cr.accuracy)
            # print(cr.balanced_accuracy)
            # print(cr.auroc_score)
            # print(cr.log_loss)