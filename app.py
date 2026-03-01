import joblib

import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model


# Load model and preprocessor once
@st.cache_resource
def load_co2_model(path):
    return load_model(path)


@st.cache_resource
def load_preprocessor(path):
    return joblib.load(path)


model = load_co2_model("co2_model.keras")
preprocessor = load_preprocessor("preprocessor.pkl")

Vehicle_classes = [
    "SUV - SMALL",
    "MID-SIZE",
    "COMPACT",
    "SUV - STANDARD",
    "SUBCOMPACT",
    "PICKUP TRUCK - STANDARD",
    "FULL-SIZE",
    "TWO-SEATER",
    "MINICOMPACT",
    "PICKUP TRUCK - SMALL",
    "STATION WAGON - SMALL",
    "VAN - PASSENGER",
    "SPECIAL PURPOSE VEHICLE",
    "MINIVAN",
    "STATION WAGON - MID-SIZE",
    "VAN - CARGO",
]

transmission_map = {
    "Automatic": "A",
    "Automated manual": "AM",
    "Automatic with select shift": "AS",
    "Continuously variable": "AV",
    "Manual": "M",
}

fuel_to_code = {
    "Regular gasoline": "X",
    "Premium gasoline": "Z",
    "Diesel": "D",
    "Ethanol (E85)": "E",
    "Natural gas": "N",
}

code_to_fuel = {v: k for k, v in fuel_to_code.items()}

# User Interface

st.set_page_config(page_title="Vehicle CO₂ Emission Predictor", layout="centered")

st.title("Vehicle CO₂ Emission Predictor")

with st.form(key="vehicle_form"):
    vehicle_class = st.selectbox("Vehicle Class", Vehicle_classes)

    engine_size = st.number_input(
        "Engine Size (L)", min_value=0.0, max_value=10.0, value=3.0
    )
    cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4)

    transmission_type = st.selectbox("Transmission Type", list(transmission_map.keys()))
    transmission_gears = st.number_input(
        "Number of gears", min_value=3, max_value=10, value=6
    )

    fuel_type_name = st.selectbox("Fuel Type", list(fuel_to_code.keys()))

    fuel_consumption = st.number_input(
        "Fuel Consumption (L/100km)", min_value=0.0, max_value=50.0, value=10.0
    )

    submit_button = st.form_submit_button(label="Predict CO₂")


# Prediction logic
if submit_button:
    # Convert Transmission to model code
    transmission_code = f"{transmission_map[transmission_type]}{transmission_gears}"

    # Convert fuel name to code
    fuel_code = fuel_to_code[fuel_type_name]

    # Create user data DataFrame
    user_data = pd.DataFrame(
        [
            {
                "Vehicle.Class": vehicle_class,
                "Engine.Size.L.": engine_size,
                "Cylinders": cylinders,
                "Transmission": transmission_code,
                "Fuel.Type": fuel_code,
                "Fuel.Consumption.Comb..L.100.km.": fuel_consumption,
            }
        ]
    )

    user_data_preprocessed = preprocessor.transform(user_data)

    # Predict CO2
    co2_pred = model.predict(user_data_preprocessed)

    # @TODO: Compare emission to other cars within the given class
    st.success(f"Predicted CO₂ emission: {co2_pred[0][0]:.0f} g/km")

    # @TODO: Provide user with car recomendation from the dataset
