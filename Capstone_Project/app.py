import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- Loading Saved Models and Encoders ---
try:
    model = pickle.load(open('crop_yield_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le_area = pickle.load(open('le_area.pkl', 'rb'))
    le_item = pickle.load(open('le_item.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found! Please run the training script first.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()

# --- Feature Names
# Order: ['year', 'average_rain_fall_mm_per_year', 'avg_temp', 'pesticide_tonnes', 'area_encoded', 'item_encoded']
feature_names = ['year', 'average_rain_fall_mm_per_year', 'avg_temp', 'pesticide_tonnes', 'area_encoded', 'item_encoded']

# --- Streamlit App Interface ---
st.title('🌱 Crop Yield Prediction for Smallholder Farmers')

st.markdown("""
This app predicts crop yield (in **hg/ha**) based on environmental and agricultural factors.
Use the sidebar to input data and click 'Predict' to see the results.
""")

# --- User Inputs ---
st.sidebar.header('Input Features')

# Get options from encoders
try:
    area_options = list(le_area.classes_)
    item_options = list(le_item.classes_)
except Exception:
    st.sidebar.error("Error reading encoder classes.")
    area_options = ["Error"]
    item_options = ["Error"]

year = st.sidebar.number_input('Year', min_value=2010, max_value=2030, value=2024)
avg_temp = st.sidebar.slider('Average Temperature (°C)', min_value=-10.0, max_value=45.0, value=20.0, step=0.5)
rainfall = st.sidebar.slider('Average Rainfall (mm/year)', min_value=0.0, max_value=8000.0, value=1200.0, step=50.0)
pesticide_tonnes = st.sidebar.slider('Pesticide Use (tonnes)', min_value=0.0, max_value=400000.0, value=50000.0, step=1000.0)
item_choice = st.sidebar.selectbox('Crop Type', options=item_options)
area_choice = st.sidebar.selectbox('Area/Country', options=area_options)


# --- Prediction Logic ---
if st.sidebar.button('Predict Crop Yield'):
    if "Error" in area_options:
        st.error("Cannot make prediction. Encoders were not loaded correctly.")
    else:
        try:
            # 1. Encode categorical inputs
            item_encoded = le_item.transform([item_choice])[0]
            area_encoded = le_area.transform([area_choice])[0]
            
            # 2. Create a DataFrame from inputs in the correct order
            input_data = pd.DataFrame(
                [[year, rainfall, avg_temp, pesticide_tonnes, area_encoded, item_encoded]],
                columns=feature_names
            )
            
            # 3. Scale the features
            input_data_scaled = scaler.transform(input_data)
            
            # 4. Make prediction
            prediction = model.predict(input_data_scaled)
            
            st.subheader(f'Predicted Crop Yield for {item_choice.title()} in {area_choice.title()}:')
            st.success(f'**{prediction[0]:.2f} hg/ha**')
            
            # --- Farmer Decision-Making Aid ---
            st.markdown("---")
            st.header("How this prediction can help:")
            st.markdown(f"""
            * **Input Allocation:** A predicted yield of **{prediction[0]:.2f} hg/ha** is a good indicator. You can use this to decide if applying more fertilizer or pesticides (like the {pesticide_tonnes:,.0f} tonnes used in this estimate) is cost-effective.
            * **Risk Assessment:** If the predicted yield is very low, it signals high risk. This could be due to factors like low rainfall ({rainfall:,.0f} mm) or high temperatures ({avg_temp}°C). This information can help in planning for potential losses or securing insurance.
            * **Market Planning:** Knowing your expected output helps you negotiate better prices and plan your market logistics in advance.
            """)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- End of App ---