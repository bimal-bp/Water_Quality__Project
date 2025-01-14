import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd

# Function to load the model from a local file
def load_model_from_file(file_path):
    return joblib.load(file_path)

# Function to process the image
def process_image(image, model):
    # Convert the image to RGB and HSV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a mask for water regions
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # --- pH Calculation ---
    blue_band = image_rgb[:, :, 2]  # Blue channel
    nir_band = np.ones_like(blue_band) * 100
    nir_band = np.clip(nir_band, 1, None)
    ratio = blue_band.astype(float) / nir_band
    pH = 8.399 - 0.827 * ratio
    pH[mask == 0] = np.nan
    average_pH = np.nanmean(pH)

    # --- Hue Calculation ---
    hue = hsv_image[:, :, 0]
    masked_hue = cv2.bitwise_and(hue, hue, mask=mask)
    average_hue = np.nanmean(masked_hue[mask != 0])

    # --- Floating Algal Index (FUI) Calculation ---
    green_band = image_rgb[:, :, 1]
    fui = (green_band.astype(float) - blue_band) / (green_band + blue_band + 1e-5)
    fui[mask == 0] = np.nan
    average_fui = np.nanmean(fui)

    # --- Dissolved Oxygen (DO) Calculation ---
    red_band = image_rgb[:, :, 0]
    do_ratio = blue_band.astype(float) / (red_band + 1e-5)
    DO = 2.0 * do_ratio + 5.0
    DO[mask == 0] = np.nan
    average_DO = np.nanmean(DO)

    # Prepare the data for model prediction
    input_data = pd.DataFrame({
        'pH': [average_pH],
        'FUI': [average_fui],
        'DO': [average_DO]
    })

    # Predict potability
    potability_prediction = model.predict(input_data.to_numpy())

    return average_pH, average_hue, average_fui, average_DO, potability_prediction[0]

# Function to assess water quality
def assess_water_quality(pH, hue, FUI, DO, potability):
    drinking_water = (6.5 <= pH <= 8.5 and DO >= 7)
    irrigation_water = (6.0 <= pH <= 8.5)
    bathing_water = (6.5 <= pH <= 8.5 and DO >= 5)
    wildlife_support = (6.5 <= pH <= 8.5 and DO >= 4)
    post_treatment_for_drinking = (6.5 <= pH <= 8.5 and DO >= 7)

    return {
        "pH": round(pH, 2),
        "Hue": round(hue, 2),
        "FUI": round(FUI, 2),
        "DO": round(DO, 2),
        "Potability Prediction": "Safe" if potability == 1 else "Not Safe",
        "Drinking Status": "Safe for Drinking" if drinking_water else "Not Safe for Drinking",
        "Irrigation Status": "Safe for Irrigation" if irrigation_water else "Not Safe for Irrigation",
        "Bathing Status": "Safe for Bathing" if bathing_water else "Not Safe for Bathing",
        "Wildlife Support": "Supports Wildlife" if wildlife_support else "Does not Support Wildlife",
        "Post Treatment Drinking Status": "Safe for Drinking after Treatment" if post_treatment_for_drinking else "Needs Treatment for Drinking"
    }

# Streamlit App
def main():
    if "page" not in st.session_state:
        st.session_state.page = "user_info"

    def set_page(page_name):
        st.session_state.page = page_name

    st.sidebar.title("Navigation")
    if st.session_state.page == "user_info":
        st.sidebar.button("User Info", disabled=True)
        st.sidebar.button("Upload Image", on_click=lambda: set_page("upload_image"))
        st.sidebar.button("Results", on_click=lambda: set_page("results"))

        st.title("User Information")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        if st.button("Next"):
            st.session_state.user_info = {"name": name, "age": age, "gender": gender}
            set_page("upload_image")

    elif st.session_state.page == "upload_image":
        st.sidebar.button("User Info", on_click=lambda: set_page("user_info"))
        st.sidebar.button("Upload Image", disabled=True)
        st.sidebar.button("Results", on_click=lambda: set_page("results"))

        st.title("Upload Image for Analysis")
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if uploaded_image and st.button("Analyze"):
            st.session_state.image = uploaded_image
            set_page("results")

    elif st.session_state.page == "results":
        st.sidebar.button("User Info", on_click=lambda: set_page("user_info"))
        st.sidebar.button("Upload Image", on_click=lambda: set_page("upload_image"))
        st.sidebar.button("Results", disabled=True)

        st.title("Water Quality Analysis Results")
        if "user_info" in st.session_state:
            user_info = st.session_state.user_info
            st.write(f"Name: {user_info['name']}")
            st.write(f"Age: {user_info['age']}")
            st.write(f"Gender: {user_info['gender']}")
        if "image" in st.session_state:
            image = np.asarray(bytearray(st.session_state.image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            model = load_model_from_file('model.pkl')
            average_pH, average_hue, average_fui, average_DO, potability_prediction = process_image(image, model)
            assessment = assess_water_quality(average_pH, average_hue, average_fui, average_DO, potability_prediction)

            st.subheader("Results")
            for key, value in assessment.items():
                st.write(f"{key}: {value}")

if __name__ == "__main__":
    main()
