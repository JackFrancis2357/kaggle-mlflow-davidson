from sklearn.ensemble import RandomForestRegressor
import mlflow


def create_model():
    model = RandomForestRegressor()
    return model


def load_model_from_mlflow():
    print("Loading mlflow model based on specific run id")
    model = mlflow.sklearn.load_model("runs:/550312c0b91d410c9dfcdbf515b783f5/model/")
    return model


def load_best_model_from_mlflow():
    print("Loading mlflow model based on best metric")
    client = mlflow.tracking.MlflowClient()
    current_experiment = dict(mlflow.get_experiment_by_name("Random Forest Regression Model"))
    experiment_id = current_experiment["experiment_id"]
    runs = client.search_runs(experiment_ids=experiment_id, order_by=["metrics.val_mae ASC"], max_results=1)
    best_run = runs[0]
    best_model_run_id = best_run.info.run_id
    model = mlflow.sklearn.load_model(f"runs:/{best_model_run_id}/model/")
    return model
