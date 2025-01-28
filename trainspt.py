import os
import joblib
import hopsworks
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Login to Hopsworks
    project = hopsworks.login(api_key_value="EP5aAsjdPutNPjHf.qXzpVQ2wrS8dHURwxxJMggYsRgWHpy42SN2CqvSB5xdHGOdqZoezwioQU9tqj4Cc")
    fs = project.get_feature_store(name='aqipredictormz')

    # Retrieve data from feature group
    feature_group = fs.get_feature_group("aqi_features")
    data = feature_group.read()

    # Data Preparation
    features = data.drop(['aqi', 'timestamp'], axis=1)  # Drop target and timestamp
    target = data['aqi']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Initialize and train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_imputed, y_train)

    # Predictions and evaluation for Linear Regression
    y_train_pred = lr_model.predict(X_train_imputed)
    y_test_pred = lr_model.predict(X_test_imputed)
    lr_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    lr_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    lr_r2 = r2_score(y_test, y_test_pred)

    print(f"Linear Regression -> Train RMSE: {lr_train_rmse:.4f}, Test RMSE: {lr_test_rmse:.4f}, R²: {lr_r2:.4f}")

    # Train and evaluate Random Forest Regressor
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf_model = RandomForestRegressor(random_state=42)
    grid_rf = GridSearchCV(rf_model, rf_params, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_rf.fit(X_train_imputed, y_train)
    best_rf_model = grid_rf.best_estimator_
    rf_train_pred = best_rf_model.predict(X_train_imputed)
    rf_test_pred = best_rf_model.predict(X_test_imputed)
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    rf_r2 = r2_score(y_test, rf_test_pred)

    print("\nRandom Forest Best Model:")
    print(f"Train RMSE: {rf_train_rmse:.4f}, Test RMSE: {rf_test_rmse:.4f}, R²: {rf_r2:.4f}")

    # Train and evaluate Gradient Boosting Regressor
    gbr_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    gbr_model = GradientBoostingRegressor(random_state=42)
    grid_gbr = GridSearchCV(gbr_model, gbr_params, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_gbr.fit(X_train_imputed, y_train)
    best_gbr_model = grid_gbr.best_estimator_
    gbr_train_pred = best_gbr_model.predict(X_train_imputed)
    gbr_test_pred = best_gbr_model.predict(X_test_imputed)
    gbr_train_rmse = np.sqrt(mean_squared_error(y_train, gbr_train_pred))
    gbr_test_rmse = np.sqrt(mean_squared_error(y_test, gbr_test_pred))
    gbr_r2 = r2_score(y_test, gbr_test_pred)

    print("\nGradient Boosting Regressor Best Model:")
    print(f"Train RMSE: {gbr_train_rmse:.4f}, Test RMSE: {gbr_test_rmse:.4f}, R²: {gbr_r2:.4f}")

    # Save models to the model registry
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Linear Regression
    lr_model_path = os.path.join(model_dir, "linear_regression_model.pkl")
    joblib.dump(lr_model, lr_model_path)
    
    # Random Forest
    rf_model_path = os.path.join(model_dir, "random_forest_model.pkl")
    joblib.dump(best_rf_model, rf_model_path)

    # Gradient Boosting
    gbr_model_path = os.path.join(model_dir, "gradient_boosting_model.pkl")
    joblib.dump(best_gbr_model, gbr_model_path)

    # Register models in Hopsworks
    mr = project.get_model_registry()

    for model_name, model_path in zip([
        "linear_regression_model", "random_forest_model", "gradient_boosting_model"],
        [lr_model_path, rf_model_path, gbr_model_path]):
        model = mr.python.create_model(
            name=model_name,
            description=f"{model_name} for AQI prediction"
        )
        model.save(model_path)
        print(f"Registered and uploaded model: {model_name}")

if __name__ == "__main__":
    main()
