import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Drop rows with missing values
    df.dropna(subset=['Rating', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Votes'], inplace=True)

    # Convert 'Duration' (e.g., '109 min' → 109)
    df['Duration'] = df['Duration'].str.replace(' min', '', regex=True).astype(float)

    # Convert 'Votes' (e.g., '1,086' → 1086)
    df['Votes'] = df['Votes'].str.replace(',', '').astype(float)

    # Combine actors into a single string
    df['Actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].apply(lambda x: ', '.join(x), axis=1)

    # Select required columns
    df = df[['Genre', 'Director', 'Actors', 'Duration', 'Votes', 'Rating']]

    return df

# ✅ Add missing `split_data` function
def split_data(df):
    X = df.drop(columns=['Rating'])  # Features
    y = df['Rating']  # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)
