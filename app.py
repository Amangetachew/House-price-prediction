from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
import joblib  # Replaced pickle with joblib

app = FastAPI()

# Load data
df = pd.read_csv("train.csv")

# Missing Values Handling
for column in df.columns:
   if df[column].isnull().any():
       if df[column].dtype in ['int64', 'float64']:
           df[column] = df[column].fillna(df[column].mean())
       elif df[column].dtype == 'object':
           df[column] = df[column].fillna(df[column].mode()[0])

# Outliers removal for SalePrice
mean_price = df['SalePrice'].mean()
std_price = df['SalePrice'].std()
df = df[(df['SalePrice'] >= mean_price - 4 * std_price) & (df['SalePrice'] <= mean_price + 4 * std_price)]

# Label Encoding for categorical columns
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
for feature in categorical_features:
   encoder = LabelEncoder()
   df[feature] = encoder.fit_transform(df[feature])

# Feature Transformation
df['LotArea'], lam = boxcox(df['LotArea'] + 1)

# Feature Engineering
df['HouseAge'] = 2023 - df['YearBuilt']  # Assuming the year is 2023
df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]

# Scaling numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for feature in numerical_features:
   scaler = StandardScaler()
   df[feature] = scaler.fit_transform(df[[feature]])

# Selecting essential features for prediction (minimizing the input)
X = df[['LotArea', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'HouseAge', 'TotalArea']]  # Reduced feature set
y = df['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
model = grid_search.best_estimator_

# Save the model using joblib
joblib.dump(model, 'model.pkl')

# Load the model from the file using joblib
model = joblib.load('model.pkl')

# Define the InputData class to handle user input
class InputData(BaseModel):
    LotArea: float
    OverallQual: float
    GrLivArea: float
    TotalBsmtSF: float
    HouseAge: float
    TotalArea: float

@app.post("/predict")
async def predict_price(input_data: InputData):
    # Prepare input data for prediction
    input_df = pd.DataFrame([input_data.dict()])
    
    # Ensure that the input features are scaled in the same way as the training data
    scaler = StandardScaler()
    input_df_scaled = scaler.fit_transform(input_df)

    # Get prediction from the model
    prediction = model.predict(input_df_scaled)
    
    # Return the prediction
    return {"prediction": prediction[0]}
