# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

model = joblib.load("model.pkl")

st.title("🏠 House Price Predictor")
st.markdown("Enter house size to estimate its price")

# User input
sqft = st.slider("Size in Square Feet", 500, 4000, 1500, step=50)

# Predict and visualize
if st.button("Predict Price"):
    # Create a DataFrame with the correct column name
    input_df = pd.DataFrame([[sqft]], columns=["square_feet"])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {prediction:,.2f}")
    
    # Visualize regression line and the prediction
    df = pd.DataFrame({
        'square_feet': np.linspace(500, 4000, 100)
    })
    df['price'] = model.predict(df[['square_feet']])
    
    fig = px.line(df, x='square_feet', y='price', title="House Size vs Price")
    fig.add_scatter(x=[sqft], y=[prediction], mode='markers',
                    marker=dict(color='red', size=12), name='Your Prediction')
    st.plotly_chart(fig)

