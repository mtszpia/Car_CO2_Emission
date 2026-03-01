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


@st.cache_data
def load_and_process_dataset(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Fuel.Consumption.Comb..mpg."])
    avg_co2_by_class = df.groupby("Vehicle.Class")["CO2.Emissions.g.km."].mean()
    return df, avg_co2_by_class


model = load_co2_model("co2_model.keras")
preprocessor = load_preprocessor("preprocessor.pkl")
cars_dataset, avg_co2_by_class = load_and_process_dataset("CO2_Emissions.csv")

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
    co2_pred = co2_pred[0][0]

    # Display feedback
    co2_avg = avg_co2_by_class[vehicle_class]

    st.markdown("### Predicted CO₂ emission")
    if co2_pred < co2_avg:
        st.markdown(
            f"""
            ## ✅ {co2_pred:.0f} g/km
            **Lower than the average for {vehicle_class} vehicles!**
            """,
        )
    else:
        st.markdown(
            f"""
            ## ⚠️ {co2_pred:.0f} g/km
            **Higher than the average for {vehicle_class} vehicles!**
            """,
        )

    filtered_vehicles = cars_dataset[
        (cars_dataset["Vehicle.Class"] == vehicle_class)
        & (cars_dataset["CO2.Emissions.g.km."] < min(co2_pred * 0.9, co2_avg))
    ]

    # Choose few cars to recommend
    n_max = 6
    recommendations = filtered_vehicles.sample(n=min(n_max, len(filtered_vehicles)))

    st.subheader(
        "Vehicles recommendations in your car's category with lower CO₂ emission"
    )

    if not recommendations.empty:
        cols = st.columns(2)  # 2 cards per row

        for i, (_, row) in enumerate(recommendations.iterrows()):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"### {row['Make']} {row['Model']}")
                    st.metric("CO₂ emission", f"{row['CO2.Emissions.g.km.']} g/km")
                    st.markdown(f"**Engine**: {row['Engine.Size.L.']} L")
                    st.markdown(f"**Fuel type**: {code_to_fuel[row['Fuel.Type']]}")
                    st.markdown("**Fuel consumption** (L/100km)")
                    st.markdown(
                        f"""
                        | Combined | City | Highway |
                        |----------|------|---------|
                        | {row['Fuel.Consumption.Comb..L.100.km.']} | {row['Fuel.Consumption.City..L.100.km.']} | {row['Fuel.Consumption.Hwy..L.100.km.']} |
                        """
                    )
    else:
        st.info("We did not find any. Your car is pretty efficient 🙂")
