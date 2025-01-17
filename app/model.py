import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
import pickle

def preprocess_data(data):
    # data = pd.read_csv(path)
    # To-do for future scope we can save label encode in pickle and use same for prediction
    label_encoder = LabelEncoder()
    data["sectorName"] = label_encoder.fit_transform(data["sectorName"])
    data["stateDescription"] = label_encoder.fit_transform(data["stateDescription"])
    data.drop(["customers", "revenue", "sales"], axis=1, inplace=True)
    return data


def train_model(data):
    X = data.drop(["price"], axis=1)
    y = data["price"]
    param_grid = {
        "n_estimators": [50, 100, 150, 200, 250]  # Adjust this range as needed
    }
    rf = RandomForestRegressor(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X, y)

    best_n_estimators = grid_search.best_params_["n_estimators"]

    best_rf = RandomForestRegressor(n_estimators=best_n_estimators, random_state=42)
    best_rf.fit(X, y)

    y_pred = best_rf.predict(X)

    mse = mean_squared_error(y, y_pred)
    print("Best number of estimators:", best_n_estimators)
    print("Mean Squared Error (MSE):", mse)
    # with open("./model.pkl", "wb") as model_file:
    #     pickle.dump(best_rf, model_file)
    import pdb; pdb.set_trace()
    import bz2
    def save_bz2_model(model, filename="./model.pkl.bz2"):
        with bz2.BZ2File(filename, "wb") as f:
            pickle.dump(model, f)
    save_bz2_model(best_rf, filename="./model.pkl.bz2")
    import os
    # import gzip
    # import shutil
    # with open("./model.pkl", "rb") as f_in, gzip.open("./model.pkl.gz", "wb") as f_out:
    #     shutil.copyfileobj(f_in, f_out)

    # os.remove("./model.pkl")


def predict_model(data):
    import bz2
    def load_bz2_model(filename="./model.pkl.bz2"):
        with bz2.BZ2File(filename, "rb") as f:
            return pickle.load(f)

    model = load_bz2_model()

    X = data.drop(["price"], axis=1)
    y_pred = model.predict(X)
    data["price"] = y_pred
    return data
