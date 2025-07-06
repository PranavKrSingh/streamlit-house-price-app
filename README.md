# 🏠 House Price Prediction Web App

This project is a **Streamlit-based web application** that predicts house prices using a trained machine learning model. It allows users to input property details and instantly get a price prediction. The app also provides a simple visualization of model behavior.

---

## 🚀 Live Demo

🔗 [Click here to view the live app](https://app-house-price-app-t4aysamkxozovbat3xjva2.streamlit.app/)

---

## 📦 Features

- 📊 Predict house prices based on user input
- 🧠 Trained linear regression model
- 💾 Real-time prediction using saved model (`model.pkl`)
- 📉 Simple data visualization for better interpretability
- 🌐 Web deployment via Streamlit Cloud

---

## 📁 Project Structure

```

streamlit-house-price-app/
├── app.py               # Streamlit app script
├── train\_model.py       # Python script to train and save model
├── model.pkl            # Trained machine learning model
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

````

---

## ⚙️ Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly (for visualization)

---

## 🛠 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/PranavKrSingh/streamlit-house-price-app.git
   cd streamlit-house-price-app
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional if `model.pkl` already exists):

   ```bash
   python train_model.py
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 📌 Author

**Pranav Kumar Singh**
Data Science Intern – Celebal Technologies
[GitHub Profile](https://github.com/PranavKrSingh)

---

## 📃 License

This project is for educational and internship use.

---

## 🙌 Acknowledgements

Special thanks to:

* [Celebal Technologies](https://www.celebaltech.com/) for the internship opportunity.
* Mentors and trainers for their valuable guidance.
