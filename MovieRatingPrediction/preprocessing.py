import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(df.columns)  # Debugging
    
    df.dropna(subset=['Rating', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Year', 'Duration', 'Votes'], inplace=True)
    
    df['Actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].apply(lambda x: ', '.join(x), axis=1)
    
    df = df[['Genre', 'Director', 'Actors', 'Year', 'Duration', 'Votes', 'Rating']]
    
    # Standardize numerical columns
    scaler = StandardScaler()
    df[['Duration', 'Votes']] = scaler.fit_transform(df[['Duration', 'Votes']])
    
    return df

def split_data(df):
    X = df.drop(columns=['Rating'])
    y = df['Rating']
    return train_test_split(X, y, test_size=0.2, random_state=42)
