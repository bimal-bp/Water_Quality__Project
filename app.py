aimport streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd

# Function to load the model from a local file
adef load_model_from_file(file_path):
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
    blue_band = image_rgb[:, :, 2]
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
    a, b = 2.0, 5.0
    DO = a * do_ratio + b
    DO[mask == 0] = np.nan
    average_DO = np.nanmean(DO)

    # --- Hardness Calculation ---
    hardness = red_band.astype(float) * 0.05
    hardness[mask == 0] = np.nan
    average_hardness = np.nanmean(hardness)

    # --- Turbidity Calculation ---
    value_channel = hsv_image[:, :, 2]
    turbidity = np.std(value_channel[mask != 0])

    # --- Organic Carbon Calculation ---
    organic_carbon = green_band.astype(float) * 0.03
    organic_carbon[mask == 0] = np.nan
    average_organic_carbon = np.nanmean(organic_carbon)

    # Prepare the data for model prediction
    input_data = pd.DataFrame({
        'pH': [average_pH],
        'Hardness': [average_hardness],
        'Organic_carbon': [average_organic_carbon],
        'Turbidity': [turbidity]
    })

    potability_prediction = model.predict(input_data.to_numpy())
    return (average_pH, average_hue, average_fui, average_DO, average_hardness,
            turbidity, average_organic_carbon, potability_prediction[0])

# Function to assess water quality
def assess_water_quality(pH, hue, FUI, DO, hardness, turbidity, organic_carbon, potability):
    drinking_water = (6.5 <= pH <= 8.5 and DO >= 7 and turbidity <= 3)
    irrigation_water = (6.0 <= pH <= 8.5)
    bathing_water = (6.5 <= pH <= 8.5 and DO >= 5 and turbidity <= 3)
    wildlife_support = (6.5 <= pH <= 8.5 and DO >= 4)
    post_treatment_for_drinking = (6.5 <= pH <= 8.5 and DO >= 7 and turbidity == 0)

    return {
        "pH": round(pH, 2),
        "Hue": round(hue, 2),
        "FUI": round(FUI, 2),
        "DO": round(DO, 2),
        "Hardness": round(hardness, 2),
        "Turbidity": round(turbidity, 2),
        "Organic Carbon": round(organic_carbon, 2),
        "Potability Prediction": "✅ Safe" if potability == 1 else "❌ Not Safe",
        "Drinking Status": "✅ Safe for Drinking" if drinking_water else "❌ Not Safe for Drinking",
        "Irrigation Status": "✅ Safe for Irrigation" if irrigation_water else "❌ Not Safe for Irrigation",
        "Bathing Status": "✅ Safe for Bathing" if bathing_water else "❌ Not Safe for Bathing",
        "Wildlife Support": "✅ Supports Wildlife" if wildlife_support else "❌ Does not Support Wildlife",
        "Post Treatment Drinking Status": "✅ Safe for Drinking after Treatment" if post_treatment_for_drinking else "❌ Needs Treatment for Drinking"
    }

# Streamlit interface
st.title("Water Quality Assessment Application")

# Page navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "User Info"

page = st.session_state.current_page

if st.sidebar.button("Go to User Info"):
    st.session_state.current_page = "User Info"
if st.sidebar.button("Go to Image Input"):
    st.session_state.current_page = "Image Input"
if st.sidebar.button("Go to Results"):
    st.session_state.current_page = "Results"

if page == "User Info":
    st.header("Step 1: Enter User Information")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    if st.button("Proceed to Step 2"):
        st.session_state["user_info"] = {"Name": name, "Age": age, "Gender": gender}
        st.session_state.current_page = "Image Input"

elif page == "Image Input":
    st.header("Step 2: Upload Image for Analysis")
    if "user_info" not in st.session_state:
        st.warning("Please complete Step 1 first.")
    else:
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if st.button("Analyze") and uploaded_image:
            model = load_model_from_file('model.pkl')
            image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            results = process_image(image, model)
            st.session_state["results"] = results
            st.session_state.current_page = "Results"

elif page == "Results":
    st.header("Step 3: Results")
    if "results" not in st.session_state:
        st.warning("Please complete Steps 1 and 2 first.")
    else:
        (average_pH, average_hue, average_fui, average_DO, average_hardness,
         turbidity, average_organic_carbon, potability_prediction) = st.session_state["results"]
        assessment = assess_water_quality(average_pH, average_hue, average_fui, average_DO, average_hardness, turbidity, average_organic_carbon, potability_prediction)

        st.subheader("Water Quality Assessment Results")
        st.markdown(f"**Name**: {st.session_state['user_info']['Name']}")
        st.markdown(f"**Age**: {st.session_state['user_info']['Age']}")
        st.markdown(f"**Gender**: {st.session_state['user_info']['Gender']}")
        st.markdown(f"**pH**: {assessment['pH']}")
        st.markdown(f"**Hue**: {assessment['Hue']}")
        st.markdown(f"**FUI**: {assessment['FUI']}")
        st.markdown(f"**DO**: {assessment['DO']}")
        st.markdown(f"**Potability Prediction**: {assessment['Potability Prediction']}")
        st.markdown(f"**Drinking Status**: {assessment['Drinking Status']}")
        st.markdown(f"**Irrigation Status**: {assessment['Irrigation Status']}")
        st.markdown(f"**Bathing Status**: {assessment['Bathing Status']}")
        st.markdown(f"**Wildlife Support**: {assessment['Wildlife Support']}")
        st.markdown(f"**Post Treatment Drinking Status**: {assessment['Post Treatment Drinking Status']}")
