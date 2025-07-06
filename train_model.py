# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Sample dataset (you can replace this with your own dataset)
data = pd.DataFrame({
    'area': [1000, 1500, 2000, 1200, 1800, 2500],
    'bedrooms': [2, 3, 3, 2, 4, 4],
    'bathrooms': [1, 2, 2, 1, 3, 3],
    'stories': [1, 2, 2, 1, 3, 2],
    'parking': [1, 1, 2, 0, 2, 3],
    'mainroad': ['yes', 'yes', 'yes', 'no', 'yes', 'yes'],
    'guestroom': ['no', 'yes', 'no', 'no', 'yes', 'yes'],
    'furnishingstatus': ['semi-furnished', 'furnished', 'unfurnished', 'semi-furnished', 'furnished', 'furnished'],
    'price': [3000000, 5000000, 6000000, 3200000, 7500000, 9000000]
})

# One-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=['furnishingstatus', 'mainroad', 'guestroom'], drop_first=True)

X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and feature columns
joblib.dump(model, 'model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("✅ Model trained and saved successfully as model.pkl and model_columns.pkl")
