from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def create_pipeline(df):
    categorical_features = ['Genre']
    
    # Frequency encoding for 'Director' and 'Actors'
    df['Director_freq'] = df.groupby('Director')['Director'].transform('count')
    df['Actors_freq'] = df.groupby('Actors')['Actors'].transform('count')
    
    transformer = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')
    
    return Pipeline(steps=[('preprocessor', transformer)])
