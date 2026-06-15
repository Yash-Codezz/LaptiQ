import time

import numpy as np

import pandas as pd

from src.preprocess import preprocess

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, PowerTransformer, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

import joblib


df = pd.read_csv("data/Laptop_Prices.csv")

data = preprocess(df)


X_input_cols = data.drop(columns=['Price'])
y_output_col = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X_input_cols, y_output_col, test_size=0.2, random_state=69)


ordinal_cols = ['GPU_Tier', 'CPU_Series', 'CPU_Segment', 'CPU_Generation']
onehot_cols = ['Brand', 'Laptop_Type', 'CPU_Brand', 'GPU_Type', 'OS']
log_transform_columns = ['RAM', 'Storage']
yeo_johnson_cols = ['Weight', 'Pixel_Per_Inch']
scaling_cols = ['CPU_Cores', 'GPU_VRAM', 'Laptop_Age']

ordinal_branch = Pipeline(steps=[
    ('ordinal_encoding', OrdinalEncoder(categories=[['low', 'mid', 'high'], ['low', 'mid', 'high'], ['low', 'mid', 'high'], ['modern', 'latest']]))
])

onehot_branch = Pipeline(steps=[
    ('one_hot_encoding', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

log_transform = Pipeline(steps=[
    ('log_transformation', FunctionTransformer(np.log1p))
])

power_transform = Pipeline(steps=[
    ('yeo_johnson_transformation', PowerTransformer(method='yeo-johnson'))
])

scaling = Pipeline(steps=[
    ('robust_sclaing', RobustScaler())
])

preprocessing = ColumnTransformer(transformers=[
    ('oridinal', ordinal_branch, ordinal_cols),
    ('one_hot', onehot_branch, onehot_cols),
    ('log', log_transform, log_transform_columns),
    ('yeo_johnson', power_transform, yeo_johnson_cols),
    ('robust_scaler', scaling, scaling_cols)
], remainder='passthrough')

final_pipeline = Pipeline(steps=[
    ('pipeline', preprocessing),
    ('laptiQ', XGBRegressor(random_state=69))
])

laptiQ = TransformedTargetRegressor(
    regressor=final_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

param_grid = {
    'regressor__laptiQ__n_estimators' : [100, 200, 300],
    'regressor__laptiQ__max_depth' : [3, 4, 5],
    'regressor__laptiQ__learning_rate' : [0.01, 0.05, 0.1],
    'regressor__laptiQ__subsample' : [0.8, 1.0],
    'regressor__laptiQ__colsample_bytree' : [0.8, 1.0]
}
grid = GridSearchCV(
    estimator=laptiQ,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)


start = time.time()
grid.fit(X_train, y_train)
print(f"\nTime Taken For Training {(time.time() - start):.3f}s")
print("\nModel Trained Succesfully!")


print(f"\nBest Parameters : {grid.best_params_}")
print(f"\nr2 Score : {grid.best_score_ * 100:.2f}%")


model = grid.best_estimator_
y_pred = grid.predict(X_test)


print(f"\n{'-'*30}Test Set Results{'-'*30}")
print(f"\nr2 Score: {(r2_score(y_test, y_pred)*100):.2f}%")

print(f"\nMean Absolute Error: ₹{mean_absolute_error(y_test, y_pred):.2f}")
def mae_percentage(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    return (mae / np.mean(y_test)) * 100 
print(f"Extreme Gradient Boosting Model is off by {mae_percentage(y_test, y_pred):.2f}%")

print(f"\nRoot Mean Square Error: ₹{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

def adjusted_r2(test_data, predict_data, n, k):
    r2 = r2_score(test_data, predict_data)
    return 1 - (((1 - r2) * (n - 1)) / (n - 1 - k))
print(f"\nAdjusted r2 score: {(adjusted_r2(y_test, y_pred, n = X_train.shape[0], k = X_train.shape[1])*100):.2f}%")


joblib.dump(model, "model/LaptiQ.pkl")
print("\nModel Saved Successfully!")