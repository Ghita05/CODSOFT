import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(df.columns)  # Debugging: Print column names

    # Drop rows with missing values in required columns
    df.dropna(subset=['Rating', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Votes'], inplace=True)

    # Convert 'Duration' from '109 min' -> 109
    df['Duration'] = df['Duration'].str.replace(' min', '', regex=True).astype(float)

    # Convert 'Votes' from '1,086' -> 1086 (remove commas)
    df['Votes'] = df['Votes'].str.replace(',', '').astype(float)

    # Combine actor columns into a single 'Actors' column
    df['Actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].apply(lambda x: ', '.join(x), axis=1)

    # Select relevant columns
    df = df[['Genre', 'Director', 'Actors', 'Duration', 'Votes', 'Rating']]

    # Normalize numeric columns (Duration, Votes)
    scaler = StandardScaler()
    df[['Duration', 'Votes']] = scaler.fit_transform(df[['Duration', 'Votes']])

    return df
