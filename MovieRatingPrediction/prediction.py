import joblib

def load_model(filename='model.pkl'):
    return joblib.load(filename)

def predict_rating(model, X_new):
    return model.predict(X_new)
