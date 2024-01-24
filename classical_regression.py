from aeon.datasets._data_loaders import load_regression
from aeon.regression.convolution_based import RocketRegressor
from aeon.regression.feature_based import Catch22Regressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from tsml.feature_based import FPCARegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import csv
import numpy as np

# from aeon.datasets.tser_data_lists import tser_monash

        
# This is the tser_data_lists of the import above
dataset_list = [
    'AppliancesEnergy', 
    # 'HouseholdPowerConsumption1', # May have misisng value problems
    # 'HouseholdPowerConsumption2',
    # 'BenzeneConcentration',
    # 'BeijingPM25Quality', 
    # 'BeijingPM10Quality', 
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

# Parameters
NUM_EXPERIMENTS = 1
N_JOBS = -1
RANDOM_STATE = 42

# Models
estimator = {
    'RocketRegressor': RocketRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
    'XGBoost': Catch22Regressor(random_state=RANDOM_STATE, estimator=XGBRegressor(), n_jobs=N_JOBS),
    'RandomForest': Catch22Regressor(random_state=RANDOM_STATE, estimator=RandomForestRegressor(), n_jobs=N_JOBS),
    'FPCR': FPCARegressor(n_jobs=N_JOBS),
}

results_list = [] 
for dataset_name in dataset_list:
    X_train, y_train = load_regression(dataset_name, split='train', load_no_missing=True)
    X_test, y_test = load_regression(dataset_name, split='test', load_no_missing=True)

    for current_model in estimator:
        for experiment in range(NUM_EXPERIMENTS):
            model = estimator[current_model].fit(X_train, y_train)
            preds = model.predict(X_test)

            # Score Metrics
            mae_value = mae(y_test, preds)
            mse_value = mse(y_test, preds)
            
            # Storing the Results
            result_entry = {
                'Dataset': dataset_name,
                'Model': current_model,
                'Experiment': experiment,
                'MAE': mae_value,
                'MSE': mse_value,
                'RMSE': np.sqrt(mse_value),
            }
            results_list.append(result_entry)
            print(dataset_name, current_model, experiment, mae_value, mse_value)

# Saving Results to csv
csv_file_path = '../experiments/classical_regression_results.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['Dataset', 'Model', 'Experiment', 'MAE', 'MSE', 'RMSE'])
    writer.writeheader()
    for result_entry in results_list:
        writer.writerow(result_entry)