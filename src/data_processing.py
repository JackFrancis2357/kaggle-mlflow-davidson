import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    print("Starting Loading Data")
    df = pd.read_csv("./house-prices-advanced-regression-techniques/train.csv")
    # There's a ton of columns in the original dataset, so let's reduce it down for ease of use
    columns_to_keep = [
        "LotArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "MoSold",
        "YrSold",
        "SalePrice",
    ]
    df = df[columns_to_keep]
    print("Finished Loading Data")
    return df


def do_tvt_split(df):
    print("Starting train val test split")
    # Get my X and my Y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values

    # Create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Further divide the training set into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    print("Finished train val test split")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_scaling(X_train, X_val, X_test):
    print("Starting to apply scaling")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print("Finished applying scaling")
    return X_train, X_val, X_test, scaler
