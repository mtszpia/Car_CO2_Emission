import joblib
import random
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers


SEED = 42


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load CSV File
    file_path = "CO2_Emissions.csv"
    df = pd.read_csv(file_path)

    # Separate Features & Target
    target_column = "CO2.Emissions.g.km."
    X = df.drop(columns=[target_column])
    y = df[target_column]

    """
    model_counts = len(X["Model"].value_counts())
    make_counts = len(X["Make"].value_counts())
    class_counts = len(X["Vehicle.Class"].value_counts())
    transmission_counts = len(X["Transmission"].value_counts())
    fuel_counts = len(X["Fuel.Type"].value_counts())
    print(
        f"{model_counts=}, {make_counts=}, {class_counts=}, {transmission_counts=}, {fuel_counts=}"
    )
    """

    # Model can memorize CO2 emission of specificx models.
    # To avoid overfitting Model column is dropped.

    # Another potential problem is converting Model column
    # from categorical to numerical column as there is 2053
    # unique models in the dataset

    X = X.drop(columns=["Model"])

    # Drop 'Make' column because changes in brand have a large effect on predictions
    # and the dataset may be imbalanced across brands, which could lead to biased results.
    X = X.drop(columns=["Make"])

    X = X.drop(
        columns=[
            "Fuel.Consumption.Comb..mpg.",
            "Fuel.Consumption.City..L.100.km.",
            "Fuel.Consumption.Hwy..L.100.km.",
        ]
    )

    # Identify Categorical & Numerical Columns
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    print("Categorical columns:", categorical_cols)
    print("Numerical columns:", numerical_cols)

    # Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # Fit & Transform Data
    X_train_processed = preprocessor.fit_transform(X_train)
    # test data set should use fit from the train dataset
    X_test_processed = preprocessor.transform(X_test)

    # Build TensorFlow Regression Model
    input_dim = X_train_processed.shape[1]
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),  # Regression output
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.summary()

    # Train Model
    history = model.fit(
        X_train_processed,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1,
    )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_fn = f"co2_model_{timestamp}.keras"
    model.save(model_fn)
    preprocessor_fn = f"preprocessor_{timestamp}.pkl"
    joblib.dump(preprocessor, preprocessor_fn)

    # Evaluate Model
    test_loss, test_mae = model.evaluate(X_test_processed, y_test)

    # calculate R2
    y_pred = model.predict(X_test_processed)
    r2 = r2_score(y_test, y_pred)

    output_text = (
        f"Test MSE: {test_loss:.6f}\n"
        f"Test MAE: {test_mae:.6f}\n"
        f"R^2 score: {r2:.6f}\n"
    )

    # Save evaluation to a file
    eval_fn = f"model_evaluation_{timestamp}.txt"
    with open(eval_fn, "w") as f:
        f.write(output_text)

    print(output_text)

    print(f"\nModel saved as {model_fn}")
    print(f"Preprocessor saved as {preprocessor_fn}")
    print(f"Evaluation results saved as {eval_fn}")


if __name__ == "__main__":
    main()
