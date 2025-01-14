import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import requests
from io import BytesIO

# Function to load model from a URL
def load_model_from_url(url):
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# Function to process the image
def process_image(image, model):
    # Convert the image to RGB and HSV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a mask for water regions
    lower_purple = np.array([120, 50, 50])  # Adjust based on water color
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # --- pH Calculation ---
    blue_band = image_rgb[:, :, 2]  # Blue channel
    nir_band = np.ones_like(blue_band) * 100  # Substitute NIR band with a constant value
    nir_band = np.clip(nir_band, 1, None)  # Avoid divide-by-zero
    ratio = blue_band.astype(float) / nir_band
    pH = 8.399 - 0.827 * ratio
    pH[mask == 0] = np.nan  # Mask non-water regions
    average_pH = np.nanmean(pH)

    # --- Hue Calculation ---
    hue = hsv_image[:, :, 0]  # Extract hue channel
    masked_hue = cv2.bitwise_and(hue, hue, mask=mask)
    average_hue = np.nanmean(masked_hue[mask != 0])

    # --- Floating Algal Index (FUI) Calculation ---
    green_band = image_rgb[:, :, 1]  # Green channel
    fui = (green_band.astype(float) - blue_band) / (green_band + blue_band + 1e-5)
    fui[mask == 0] = np.nan  # Mask non-water regions
    average_fui = np.nanmean(fui)

    # --- Dissolved Oxygen (DO) Calculation ---
    red_band = image_rgb[:, :, 0]  # Red channel
    do_ratio = blue_band.astype(float) / (red_band + 1e-5)  # Avoid divide-by-zero
    a, b = 2.0, 5.0
    DO = a * do_ratio + b
    DO[mask == 0] = np.nan  # Mask non-water regions
    average_DO = np.nanmean(DO)

    # --- Hardness Calculation ---
    hardness = red_band.astype(float) * 0.05  # Scaling factor for demonstration
    hardness[mask == 0] = np.nan  # Mask non-water regions
    average_hardness = np.nanmean(hardness)

    # --- Turbidity Calculation ---
    value_channel = hsv_image[:, :, 2]  # Value channel from HSV
    turbidity = np.std(value_channel[mask != 0])  # Standard deviation as a proxy for turbidity

    # --- Organic Carbon Calculation ---
    organic_carbon = green_band.astype(float) * 0.03  # Scaling factor for demonstration
    organic_carbon[mask == 0] = np.nan
    average_organic_carbon = np.nanmean(organic_carbon)

    # Prepare the data for model prediction
    input_data = pd.DataFrame({
        'pH': [average_pH],
        'Hardness': [average_hardness],
        'Organic_carbon': [average_organic_carbon],
        'Turbidity': [turbidity]
    })

    # Convert the DataFrame to numpy array (without column names) for prediction
    input_data_np = input_data.to_numpy()

    # Predict potability (use numpy array without column names)
    potability_prediction = model.predict(input_data_np)

    return average_pH, average_hue, average_fui, average_DO, average_hardness, turbidity, average_organic_carbon, potability_prediction[0]

# Function to assess water quality
def assess_water_quality(pH, hue, FUI, DO, hardness, turbidity, organic_carbon, potability):
    # Define acceptable ranges for different water uses
    drinking_water = (6.5 <= pH <= 8.5 and DO >= 7 and turbidity <= 3)
    irrigation_water = (6.0 <= pH <= 8.5)
    bathing_water = (6.5 <= pH <= 8.5 and DO >= 5 and turbidity <= 3)
    wildlife_support = (6.5 <= pH <= 8.5 and DO >= 4)
    post_treatment_for_drinking = (6.5 <= pH <= 8.5 and DO >= 7 and turbidity == 0)

    # Water quality categorization
    drinking_status = "Safe for Drinking" if drinking_water else "Not Safe for Drinking"
    irrigation_status = "Safe for Irrigation" if irrigation_water else "Not Safe for Irrigation"
    bathing_status = "Safe for Bathing" if bathing_water else "Not Safe for Bathing"
    wildlife_status = "Supports Wildlife" if wildlife_support else "Does not Support Wildlife"
    post_treatment_status = "Safe for Drinking after Treatment" if post_treatment_for_drinking else "Needs Treatment for Drinking"

    # Return the results in a dictionary
    return {
        "pH": round(pH, 2),
        "Hue": round(hue, 2),
        "FUI": round(FUI, 2),
        "DO": round(DO, 2),
        "Hardness": round(hardness, 2),
        "Turbidity": round(turbidity, 2),
        "Organic Carbon": round(organic_carbon, 2),
        "Potability Prediction": "Safe" if potability == 1 else "Not Safe",
        "Drinking Status": drinking_status,
        "Irrigation Status": irrigation_status,
        "Bathing Status": bathing_status,
        "Wildlife Support": wildlife_status,
        "Post Treatment Drinking Status": post_treatment_status
    }

# Streamlit interface
st.title("Water Quality Assessment")
st.write("Upload the image for water quality analysis.")

# File uploader for image
uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# URL for the model stored in GitHub (replace with your actual URL)
model_url = "https://github.com/bimal-bp/Water_Quality__Project/blob/main/model.pkl"

if uploaded_image:
    # Load the model from GitHub
    model = load_model_from_url(model_url)
    
    # Read the image from the uploaded file
    image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Process the image and get the water quality parameters
    average_pH, average_hue, average_fui, average_DO, average_hardness, turbidity, average_organic_carbon, potability_prediction = process_image(image, model)

    # Assess the water quality
    assessment = assess_water_quality(average_pH, average_hue, average_fui, average_DO, average_hardness, turbidity, average_organic_carbon, potability_prediction)

    # Display the results
    st.subheader("Water Quality Assessment Results")
    st.write(f"pH: {assessment['pH']}")
    st.write(f"DO: {assessment['DO']}")
    st.write(f"Hue: {assessment['Hue']:.2f}")
    st.write(f"FUI: {assessment['FUI']:.2f}")
    st.write(f"Potability Prediction: {assessment['Potability Prediction']}")
    st.write(f"Drinking Status: {assessment['Drinking Status']}")
    st.write(f"Irrigation Status: {assessment['Irrigation Status']}")
    st.write(f"Bathing Status: {assessment['Bathing Status']}")
    st.write(f"Wildlife Support: {assessment['Wildlife Support']}")
    st.write(f"Post Treatment Drinking Status: {assessment['Post Treatment Drinking Status']}")
