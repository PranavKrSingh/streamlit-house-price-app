# train_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def generate_data(n=100):
    np.random.seed(42)
    sqft = np.random.normal(1500, 400, n)
    price = sqft * 100 + np.random.normal(0, 10000, n)
    return pd.DataFrame({'square_feet': sqft, 'price': price})

# Generate and split data
df = generate_data()
X = df[['square_feet']]
y = df['price']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save model
model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
