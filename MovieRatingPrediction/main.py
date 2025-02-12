from preprocessing import load_and_preprocess_data, split_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from modeling import evaluate_model, save_model, load_model, predict
import numpy as np

def main():
    file_path = "IMDb_Movies_India.csv"
    
    df = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # Identify column types
    categorical_features = ['Genre', 'Director', 'Actors']
    numerical_features = ['Duration', 'Votes']

    # Define transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Build pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor())
    ])

    # Define hyperparameters for RandomizedSearch
    param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")

    # Evaluate the model
    error = evaluate_model(best_model, X_test, y_test)
    print(f'Model RMSE: {error}')

    # Save the model
    save_model(best_model)
    print('Model saved successfully.')

    # Load the model and make predictions on new data
    loaded_model = load_model()
    X_new = X_test  # Replace with your new data
    predictions = predict(loaded_model, X_new)
    print(f'Predictions: {predictions}')

if __name__ == "__main__":
    main()
