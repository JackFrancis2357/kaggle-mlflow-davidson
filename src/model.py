import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


def create_model(model_configs):
    if model_configs["model"] == "RandomForestRegressor":
        model = RandomForestRegressor()
    elif model_configs["model"] == "LinearRegression":
        model = LinearRegression()
    return model


def create_gridsearch_model(model_configs):
    param_grid = {
        "max_depth": [None, 5, 8, 10, 12, 15],
        "min_samples_split": [10, 12, 13, 14, 16],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": [None, "sqrt", "log2"],
    }

    if model_configs["model"] == "RandomForestRegressor":
        model = RandomForestRegressor()
    elif model_configs["model"] == "LinearRegression":
        model = LinearRegression()

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1, verbose=1
    )
    return grid_search


def load_model_from_mlflow():
    print("Loading mlflow model based on specific run id")
    model = mlflow.sklearn.load_model("runs:/f8ba0ee5524642e5a78393610840b059/model/")
    return model


def load_best_model_from_mlflow():
    print("Loading mlflow model based on best metric")
    client = mlflow.tracking.MlflowClient()
    current_experiment = dict(mlflow.get_experiment_by_name("Random Forest Regression HP Tuning Model"))
    experiment_id = current_experiment["experiment_id"]
    runs = client.search_runs(experiment_ids=experiment_id, order_by=["metrics.Validation_MAE DESC"], max_results=1)
    best_run = runs[0]
    best_model_run_id = best_run.info.run_id
    model = mlflow.sklearn.load_model(f"runs:/{best_model_run_id}/model/")
    return model
