import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(subset=['Rating', 'Genre', 'Director', 'Actors'], inplace=True)
    df = df[['Genre', 'Director', 'Actors', 'Rating']]
    return df

def split_data(df):
    X = df.drop(columns=['Rating'])
    y = df['Rating']
    return train_test_split(X, y, test_size=0.2, random_state=42)