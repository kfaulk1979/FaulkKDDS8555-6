import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


# load Hitters dataset
from ISLP import load_data
Hitters = load_data('Hitters').dropna()

# Prepare features and target
X = pd.get_dummies(Hitters.drop(columns='Salary'), drop_first=True)
y = Hitters['Salary']
feature_names = X.columns

# Standardize numeric columns
scaler = StandardScaler()
X[X.select_dtypes(include=np.number).columns] = scaler.fit_transform(X.select_dtypes(include=np.number))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Boosting 
boost_model = GBR(n_estimators=5000, learning_rate=0.01, max_depth=3, random_state=0)
boost_model.fit(X_train, y_train)
y_hat_boost = boost_model.predict(X_test)
boost_mse = mean_squared_error(y_test, y_hat_boost)

# Plot training vs test error
test_error = np.zeros_like(boost_model.train_score_)
for idx, y_ in enumerate(boost_model.staged_predict(X_test)):
    test_error[idx] = mean_squared_error(y_test, y_)

plt.figure(figsize=(8, 6))
plt.plot(boost_model.train_score_, label="Train Error")
plt.plot(test_error, label="Test Error", color='red')
plt.xlabel("Boosting Iterations")
plt.ylabel("MSE")
plt.title("Boosting: Training vs Test Error")
plt.legend()
plt.grid(True)
plt.show()

# Bagging
bag_model = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=500,
    random_state=0
)
bag_model.fit(X_train, y_train)
y_hat_bag = bag_model.predict(X_test)
bag_mse = mean_squared_error(y_test, y_hat_bag)

# Random Forest
rf_model = RF(max_features=6, random_state=0)
rf_model.fit(X_train, y_train)
y_hat_rf = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, y_hat_rf)

# Feature importance
rf_feature_imp = pd.DataFrame({
    'importance': rf_model.feature_importances_
}, index=feature_names).sort_values(by='importance', ascending=False)
print("\nTop Random Forest Feature Importances:\n", rf_feature_imp.head())

# BART
# --- BART using R from Python via rpy2 ---
pandas2ri.activate()
BART = importr('BART')

# Convert data to R objects
X_train_r = pandas2ri.py2rpy(X_train)
y_train_r = ro.FloatVector(y_train)
X_test_r = pandas2ri.py2rpy(X_test)

# Fit the BART model using wbart
bart_fit = BART.wbart(x_train=X_train_r, y_train=y_train_r, x_test=X_test_r)

# The posterior mean prediction is the column mean across all posterior draws
# Each row = one draw, each column = one test observation
bart_draws = np.array(bart_fit.rx2("yhat.test"))
bart_preds = bart_draws.mean(axis=0)

# Compute and print BART MSE
bart_mse = mean_squared_error(y_test, bart_preds)
print(f"\nBART MSE (via R wbart): {bart_mse:.2f}")

var_counts = np.array(bart_fit.rx2("varcount"))
var_importance = var_counts.sum(axis=0)
sorted_indices = np.argsort(var_importance)[::-1]
top_vars = X.columns[sorted_indices][:5]
print("Top BART variables:", top_vars.tolist())

# Linear Regression
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)
y_hat_linreg = linreg_model.predict(X_test)
linreg_mse = mean_squared_error(y_test, y_hat_linreg)

# Model Comparison
print("\nModel Comparison (Test Set MSE):")
print(f"Linear Regression MSE: {linreg_mse:.2f}")
print(f"Boosting MSE:           {boost_mse:.2f}")
print(f"Bagging MSE:            {bag_mse:.2f}")
print(f"Random Forest MSE:      {rf_mse:.2f}")
print(f"BART MSE (via R wbart): {bart_mse:.2f}")
