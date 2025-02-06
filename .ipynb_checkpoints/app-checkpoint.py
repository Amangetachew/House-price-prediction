# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from scipy.stats import boxcox
import pickle # Library to save the model

app = FastAPI()


# Load data
df = pd.read_csv("train.csv")
#Missing Values
for column in df.columns:
   if df[column].isnull().any():
       if df[column].dtype in ['int64', 'float64']:
           df[column] = df[column].fillna(df[column].mean())
       elif df[column].dtype == 'object':
           df[column] = df[column].fillna(df[column].mode()[0])
# Remove outliers in SalePrice
mean_price = df['SalePrice'].mean()
std_price = df['SalePrice'].std()
df = df[(df['SalePrice'] >= mean_price - 4 * std_price) & (df['SalePrice'] <= mean_price + 4 * std_price)]
#Label Encoding
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
for feature in categorical_features:
   encoder = LabelEncoder()
   df[feature] = encoder.fit_transform(df[feature])
# Feature Transformation
df['LotArea'], lam = boxcox(df['LotArea'] + 1)
# Feature Engineering
df['HouseAge'] = 2023 - df['YearBuilt'] #assuming the year is 2023
df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
# Scaling
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for feature in numerical_features:
   scaler = StandardScaler()
   df[feature] = scaler.fit_transform(df[[feature]])
# Split the data
X = df.drop(['SalePrice', 'Id'], axis=1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Hyperparameter Tuning
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# We save the model to a pickle file
with open('model.pkl', 'wb') as f:
   pickle.dump(model, f)

# We load the model from the file
with open('model.pkl', 'rb') as f:
   model = pickle.load(f)


class InputData(BaseModel):
 LotArea: float
 OverallQual: float
 YearBuilt: float
 GrLivArea: float
 TotalBsmtSF: float
 HouseAge: float
 TotalArea: float
 MSZoning: int
 Street: int
 Alley: int
 LotShape: int
 LandContour: int
 Utilities: int
 LotConfig: int
 LandSlope: int
 Neighborhood: int
 Condition1: int
 Condition2: int
 BldgType: int
 HouseStyle: int
 RoofStyle: int
 RoofMatl: int
 Exterior1st: int
 Exterior2nd: int
 MasVnrType: int
 MasVnrArea: float
 ExterQual: int
 ExterCond: int
 Foundation: int
 BsmtQual: int
 BsmtCond: int
 BsmtExposure: int
 BsmtFinType1: int
 BsmtFinSF1: float
 BsmtFinType2: int
 BsmtFinSF2: float
 BsmtUnfSF: float
 Heating: int
 HeatingQC: int
 CentralAir: int
 Electrical: int
 FirstFlrSF: float
 SecondFlrSF: float
 LowQualFinSF: float
 BsmtFullBath: float
 BsmtHalfBath: float
 FullBath: float
 HalfBath: float
 BedroomAbvGr: int
 KitchenAbvGr: int
 KitchenQual: int
 TotRmsAbvGrd: int
 Functional: int
 Fireplaces: int
 FireplaceQu: int
 GarageType: int
 GarageYrBlt: float
 GarageFinish: int
 GarageCars: float
 GarageArea: float
 GarageQual: int
 GarageCond: int
 PavedDrive: int
 WoodDeckSF: float
 OpenPorchSF: float
 EnclosedPorch: float
 ThreeSsnPorch: float
 ScreenPorch: float
 PoolArea: float
 PoolQC: int
 Fence: int
 MiscFeature: int
 MiscVal: float

@app.post("/predict")
async def predict_price(input_data: InputData):
 input_df = pd.DataFrame([input_data.dict()])
 prediction = model.predict(input_df)
 return {"prediction": prediction[0]}