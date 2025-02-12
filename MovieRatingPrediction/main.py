from preprocessing import load_and_preprocess_data, split_data
from feature import create_pipeline
from modeling import train_model, evaluate_model, save_model, load_model, predict

def main():
    file_path = 'IMDb_Movies_India.csv'
    df = load_and_preprocess_data(file_path)
    
    # Apply frequency encoding
    df['Director_freq'] = df.groupby('Director')['Director'].transform('count')
    df['Actors_freq'] = df.groupby('Actors')['Actors'].transform('count')
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    pipeline = create_pipeline(df)
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    model = train_model(X_train_transformed, y_train)
    error = evaluate_model(model, X_test_transformed, y_test)
    print(f'Improved Model RMSE: {error}')
    
    save_model(model)
    print('Improved model saved successfully.')

if __name__ == '__main__':
    main()
