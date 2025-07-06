
# 🏠 House Price Prediction App

A powerful web app built using **Streamlit** that allows users to **predict house prices** using a trained machine learning model.  
Supports both **manual input** and **CSV batch upload** with flexible column support and visual analytics.

🔗 [Live App](https://app-house-price-app-t4aysamkxozovbat3xjva2.streamlit.app/)  
🔗 [GitHub Repository](https://github.com/PranavKrSingh/streamlit-house-price-app)

---

## ✨ Features

✅ **User-Friendly Interface** using Streamlit  
✅ **Single House Prediction** via sidebar inputs  
✅ **Multiple Predictions** via CSV upload  
✅ **Automatic Column Mapping** for CSV inputs  
✅ **Model Evaluation Metrics** – R², MAE, RMSE  
✅ **Interactive Visualizations** (Bar, Pie, Line graphs)  
✅ **Downloadable Prediction Results**  
✅ **Example CSV format** preview for reference  
✅ Handles flexible values (e.g., any number of stories, parking spaces)  
✅ Clean UI with optional input collapse after prediction  

---

## 📊 Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas
- joblib
- plotly

---

## 🧠 Model Training

The ML model is trained on structured housing data and stored as:
- `model.pkl` → Trained model
- `model_columns.pkl` → Expected input feature columns

You can retrain the model by running:

```bash
python train_model.py
````

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/PranavKrSingh/streamlit-house-price-app.git
cd streamlit-house-price-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📁 CSV Format for Batch Prediction

Use any CSV with some or all of the following columns:

```csv
area,bedrooms,bathrooms,stories,parking,mainroad,guestroom,furnishingstatus
1200,3,2,2,1,Yes,Yes,Furnished
800,2,1,1,0,No,No,Unfurnished
```

> ✅ Columns like `sqft`, `square_feet`, `no_of_bedrooms`, etc. will be auto-renamed.
> ✅ Additional or missing columns will be handled gracefully.

📥 You can also [Download Sample CSV Here](https://github.com/PranavKrSingh/streamlit-house-price-app/blob/main/sample_input.csv)



## 🙋‍♂️ Author

**Pranav Kumar Singh**
🔗 [GitHub](https://github.com/PranavKrSingh)
📫 Feel free to connect!



