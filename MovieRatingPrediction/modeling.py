from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return mean_squared_error(y_test, predictions, squared=False)

def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)