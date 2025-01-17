from app.model import preprocess_data, train_model, predict_model
import sys
import pandas as pd
sys.path.append(".")

if __name__ == "__main__":
    data = pd.read_csv("./clean_data.csv")
    data = preprocess_data(data)
    train_model(data)
    predict_model(data)
    print("Model training and prediction completed.")