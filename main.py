import mlflow

from src.data_processing import load_data, do_tvt_split, apply_scaling
from src.model import create_model

mlflow.sklearn.autolog()

df = load_data()

X_train, X_val, X_test, y_train, y_val, y_test = do_tvt_split(df)

X_train, X_val, X_test, scaler = apply_scaling(X_train, X_val, X_test)

model = create_model()

model.fit(X_train, y_train)
