import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model and model columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction App")
st.markdown("Built with **Streamlit** | Predict house prices using a trained machine learning model.")

# Sidebar inputs
st.sidebar.header("📋 Enter House Details")

area = st.sidebar.number_input("Area (in sq ft)", min_value=100, max_value=10000, step=50, value=1000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
stories = st.sidebar.number_input("Number of Stories", min_value=1, step=1, value=2)
parking = st.sidebar.number_input("Parking Spaces", min_value=0, step=1, value=1)
mainroad = st.sidebar.selectbox("Is Property Near Main Road?", ['Yes', 'No'])
guestroom = st.sidebar.selectbox("Has Guest Room?", ['Yes', 'No'])
furnishing = st.sidebar.selectbox("Furnishing Status", ['Furnished', 'Semi-Furnished', 'Unfurnished'])

# Convert inputs to model format
input_dict = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad_yes': 1 if mainroad == 'Yes' else 0,
    'guestroom_yes': 1 if guestroom == 'Yes' else 0,
    'furnishingstatus_furnished': 1 if furnishing == 'Furnished' else 0,
    'furnishingstatus_semi-furnished': 1 if furnishing == 'Semi-Furnished' else 0
}

# Convert to DataFrame and align with model
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict button
if st.sidebar.button("🔍 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹{round(prediction):,}")
    st.balloons()

    # Input summary
    st.subheader("📊 Input Summary")
    fig = px.bar(x=input_df.columns, y=input_df.values[0], labels={'x': 'Feature', 'y': 'Value'})
    st.plotly_chart(fig)

    

# ------------------ CSV Upload Section ------------------
st.subheader("📁 Batch Prediction with CSV")

# Show sample CSV format
st.markdown("### 📄 Expected CSV Format Example:")
sample_csv = pd.DataFrame({
    'area': [1200],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'parking': [1],
    'mainroad': ['Yes'],
    'guestroom': ['No'],
    'furnishingstatus': ['Furnished']
})
st.dataframe(sample_csv)

st.markdown("👉 Make sure your uploaded CSV follows this structure.")

# Upload CSV
csv_file = st.file_uploader("Upload CSV File", type=["csv"])

if csv_file is not None:
    try:
        user_df = pd.read_csv(csv_file)

        st.write("📄 Uploaded Data Preview:")
        st.dataframe(user_df.head())

        # Rename columns if needed
        rename_map = {
            'sqft': 'area',
            'square_feet': 'area',
            'no_of_bedrooms': 'bedrooms',
            'no_of_bathrooms': 'bathrooms',
            'no_of_stories': 'stories',
            'car_parking': 'parking'
        }
        user_df.rename(columns=rename_map, inplace=True)

        # Process input data
        processed_df = pd.DataFrame()
        processed_df['area'] = user_df.get('area', 0)
        processed_df['bedrooms'] = user_df.get('bedrooms', 0)
        processed_df['bathrooms'] = user_df.get('bathrooms', 0)
        processed_df['stories'] = user_df.get('stories', 1)
        processed_df['parking'] = user_df.get('parking', 0)
        processed_df['mainroad_yes'] = user_df.get('mainroad', '').apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        processed_df['guestroom_yes'] = user_df.get('guestroom', '').apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        processed_df['furnishingstatus_furnished'] = user_df.get('furnishingstatus', '').apply(
            lambda x: 1 if str(x).lower() == 'furnished' else 0)
        processed_df['furnishingstatus_semi-furnished'] = user_df.get('furnishingstatus', '').apply(
            lambda x: 1 if str(x).lower() == 'semi-furnished' else 0)

        # Ensure all model columns exist
        for col in model_columns:
            if col not in processed_df.columns:
                processed_df[col] = 0

        processed_df = processed_df[model_columns]

        predictions = model.predict(processed_df)
        user_df['Predicted Price'] = predictions

        st.success("✅ Predictions completed!")
        st.dataframe(user_df)

        # Download button
        result_csv = user_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Predictions CSV", data=result_csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Could not process file: {e}")

# ------------------ Optional Metrics ------------------
st.subheader("📈 Model Performance (Sample Input)")

try:
    test_X = input_df
    test_y = [model.predict(test_X)[0]]
    y_pred = model.predict(test_X)

    if len(test_y) >= 2:
        r2 = r2_score(test_y, y_pred)
        mae = mean_absolute_error(test_y, y_pred)
        rmse = np.sqrt(mean_squared_error(test_y, y_pred))

        st.markdown(f"- R² Score: **{r2:.2f}**")
        st.markdown(f"- MAE: **{mae:.2f}**")
        st.markdown(f"- RMSE: **{rmse:.2f}**")
    else:
        st.info("ℹ️ Model metrics require at least 2 samples for meaningful evaluation.")

except:
    st.warning("⚠️ Model evaluation skipped due to missing test data.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("🔧 Developed by [Pranav Kumar Singh](https://github.com/PranavKrSingh) &nbsp; | &nbsp; [📁 GitHub Repo](https://github.com/PranavKrSingh/streamlit-house-price-app)")
