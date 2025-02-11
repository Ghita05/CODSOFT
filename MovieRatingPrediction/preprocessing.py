import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Specify the encoding here
    print(df.columns)  # Print the columns to debug
    df.dropna(subset=['Rating', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], inplace=True)
    df['Actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].apply(lambda x: ', '.join(x), axis=1)
    df = df[['Genre', 'Director', 'Actors', 'Rating']]
    return df

def split_data(df):
    X = df.drop(columns=['Rating'])
    y = df['Rating']
    return train_test_split(X, y, test_size=0.2, random_state=42)