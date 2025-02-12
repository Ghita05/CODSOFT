from preprocessing import load_and_preprocess_data, split_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

def main():
    file_path = "IMDb_Movies_India.csv"
    
    df = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # ✅ Identify column types
    categorical_features = ['Genre', 'Director', 'Actors']
    numerical_features = ['Duration', 'Votes']

    # ✅ Define transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # ✅ Build pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # ✅ Fit the model
    X_train_transformed = pipeline.fit_transform(X_train)
    print("Pipeline successfully applied!")

if __name__ == "__main__":
    main()
