from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_pipeline():
    categorical_features = ['Genre', 'Director', 'Actors']
    transformer = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    return Pipeline(steps=[('preprocessor', transformer)])