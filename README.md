# Car $CO_2$ Emission Prediction
A machine learning project featuring a Neural Network designed to predict vehicle $CO_2$ emissions, complete with an interactive Streamlit web interface.

## Features
- **Neural Network Core:** A Keras-based regression model trained to predict $CO_2$ emissions ($g/km$) based on vehicle specifications.
- **Integrated Pipeline:** Uses a saved ColumnTransformer to ensure data scaling and encoding are identical between training and the live UI.
- **Web UI**: Interactive Streamlit interface for real-time inference and "what-if" scenario testing.

## Getting started

1. Environment Setup

    Clone the repository and initialize your virtual environment:

    ``` bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Run the Application

    The repository includes a pre-trained model for immediate use.

    ``` bash
    streamlit run app.py
    ```

### Training Your Own Model

1. Run the training script
    ```bash
    python3 train_model.py
    ```

    **Note**: This generates three files: `co2_model_{timestamp}.keras`, `preprocessor_{timestamp}.pkl`, and a performance report.

2. Update Model Links:

    The UI (`app.py`) looks for files named `co2_model.keras` and `preprocessor.pkl`. You must rename your latest generated files to these names to see your changes in the UI.

## Model Performance

The current model achieves an $R^2$ score of ~0.997  on the test set, indicating high predictive accuracy for most vehicle classes.

## Data Source

This project uses the CO2 Emission by Vehicles database, available at https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles. It is licensed under the [ODbL](https://opendatacommons.org/licenses/odbl/1-0/) and [DbCL](https://opendatacommons.org/licenses/dbcl/1-0/).

### Data Processing Note

The dataset used in this repository is a **processed version** of the original.
