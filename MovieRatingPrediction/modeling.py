from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5  # Calculate RMSE manually
    return rmse

def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='model.pkl'):
    return joblib.load(filename)

def predict(model, X_new):
    return model.predict(X_new)