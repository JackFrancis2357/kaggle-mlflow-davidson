import mlflow
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error

from src.data_processing import load_data, do_tvt_split, apply_scaling
from src.model import create_model, create_gridsearch_model, load_model_from_mlflow, load_best_model_from_mlflow

with open("config.yml", 'r') as file:
    model_configs = yaml.safe_load(file)

df = load_data()

X_train, X_val, X_test, y_train, y_val, y_test = do_tvt_split(df)

if model_configs['scaling']:
    X_train, X_val, X_test, scaler = apply_scaling(X_train, X_val, X_test)


def calculate_mae(x, y, model):
    y_pred = model.predict(x)
    return mean_absolute_error(y, y_pred)


def get_scores(x_train, x_val, x_test, y_train, y_val, y_test, model):
    train_mae = calculate_mae(x_train, y_train, model)
    val_mae = calculate_mae(x_val, y_val, model)
    test_mae = calculate_mae(x_test, y_test, model)
    print(
        f"""
    ----------------------------------------
    Training MAE: \t {np.round(train_mae, 4)}
    Validation MAE: \t {np.round(val_mae, 4)}
    Test MAE: \t \t {np.round(test_mae, 4)}
    ----------------------------------------
    """
    )
    return train_mae, val_mae, test_mae


if model_configs['do_training']:
    mlflow.sklearn.autolog(max_tuning_runs=10)
    if model_configs['use_gridsearch']:
        model = create_gridsearch_model(model_configs)
    else:
        model = create_model(model_configs)
    mlflow.set_experiment(model_configs['experiment_name'])
    experiment = mlflow.get_experiment_by_name(model_configs['experiment_name'])

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model.fit(X_train, y_train)
        train_mae, val_mae, test_mae = get_scores(X_train, X_val, X_test, y_train, y_val, y_test, model)
        mlflow.log_metric("Train_MAE", train_mae)
        mlflow.log_metric("Validation_MAE", val_mae)
        mlflow.log_metric("Test_MAE", test_mae)
        mlflow.log_params(model_configs)
        mlflow.log_artifact('config.yml')

if model_configs['do_testing']:
    model = load_best_model_from_mlflow()
    train_mae, val_mae, test_mae = get_scores(X_train, X_val, X_test, y_train, y_val, y_test, model)
