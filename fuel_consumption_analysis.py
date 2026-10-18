import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save charts without opening GUI windows.
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    # =========================
    # 1. Read dataset
    # =========================
    project_dir = Path(__file__).resolve().parent
    file = project_dir / "data" / "Automobile.csv"
    df = pd.read_csv(file)

    print("===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== BASIC INFO =====")
    df.info()

    print("\n===== COLUMN NAMES =====")
    print(df.columns)

    print("\n===== STATISTICAL SUMMARY =====")
    print(df.describe())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    # =========================
    # 2. Cleaning dataset
    # =========================
    df_clean = df.dropna().copy()

    print("\n===== AFTER CLEANING =====")
    print("Rows before cleaning:", len(df))
    print("Rows after cleaning:", len(df_clean))
    print("Removed rows:", len(df) - len(df_clean))

    # =========================
    # 3. Unit conversion
    # =========================

    # Convert MPG to liters per 100 kilometers.
    # MPG means miles per gallon.
    # In Europe fuel consumption is usually expressed as l/100 km.
    df_clean["fuel_consumption_l_100km"] = 235.214583 / df_clean["mpg"]

    # Convert displacement from cubic inches to cubic centimeters.
    # 1 cubic inch = 16.387064 cm3.
    df_clean["displacement_cm3"] = df_clean["displacement"] * 16.387064

    print("\n===== DATA AFTER ADDING NEW COLUMNS =====")
    print(df_clean[[
        "name",
        "mpg",
        "fuel_consumption_l_100km",
        "displacement",
        "displacement_cm3",
        "horsepower",
        "weight"
    ]].head())

    print("\n===== STATISTICAL SUMMARY AFTER CONVERSION =====")
    print(df_clean[[
        "fuel_consumption_l_100km",
        "displacement_cm3",
        "horsepower",
        "weight",
        "acceleration",
        "model_year"
    ]].describe())

    # =========================
    # 4. Data visualization
    # =========================

    # Create folder for charts if it does not exist.
    OUTPUT_DIR = project_dir / "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Select numerical columns for analysis.
    numeric_columns = [
        "fuel_consumption_l_100km",
        "cylinders",
        "displacement_cm3",
        "horsepower",
        "weight",
        "acceleration",
        "model_year"
    ]

    # Histograms show the distribution of numerical features.
    df_clean[numeric_columns].hist(figsize=(12, 8), bins=20, edgecolor="black")
    plt.suptitle("Histograms of numerical features")
    plt.tight_layout()

    histogram_path = os.path.join(OUTPUT_DIR, "histograms.png")
    plt.savefig(histogram_path, dpi=200)
    plt.close()

    print("\nSaved histogram chart:", histogram_path)

    # Correlation matrix shows relationships between numerical columns.
    correlation_matrix = df_clean[numeric_columns].corr()

    print("\n===== CORRELATION MATRIX =====")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation coefficient")

    plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45, ha="right")
    plt.yticks(range(len(numeric_columns)), numeric_columns)

    # Add correlation values inside the chart.
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            plt.text(
                j,
                i,
                round(correlation_matrix.iloc[i, j], 2),
                ha="center",
                va="center",
                fontsize=8
            )

    plt.title("Correlation matrix of numerical features")
    plt.tight_layout()

    correlation_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(correlation_path, dpi=200)
    plt.close()

    print("Saved correlation matrix chart:", correlation_path)

    # Check how many cars come from each origin.
    print("\n===== ORIGIN VALUE COUNTS =====")
    print(df_clean["origin"].value_counts())

    df_clean["origin"].value_counts().plot(kind="bar", figsize=(7, 5), edgecolor="black")
    plt.xlabel("Origin")
    plt.ylabel("Number of cars")
    plt.title("Number of cars by origin")
    plt.tight_layout()

    origin_path = os.path.join(OUTPUT_DIR, "origin_counts.png")
    plt.savefig(origin_path, dpi=200)
    plt.close()

    print("Saved origin chart:", origin_path)


    # =========================
    # 5. Preparing data for machine learning models
    # =========================

    # Select features used to predict fuel consumption.
    features = [
        "cylinders",
        "displacement_cm3",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin"
    ]

    # X contains input features.
    X = df_clean[features].copy()

    # y contains the target variable.
    y = df_clean["fuel_consumption_l_100km"].copy()

    print("\n===== FEATURES BEFORE ENCODING =====")
    print(X.head())

    print("\n===== TARGET VARIABLE =====")
    print(y.head())

    # The origin column is categorical, so we need to convert it into numbers.
    # One-hot encoding creates separate columns for each origin category.
    X = pd.get_dummies(X, columns=["origin"], drop_first=True)

    print("\n===== FEATURES AFTER ONE-HOT ENCODING =====")
    print(X.head())

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    # =========================
    # Standardization decision
    # =========================

    # Standardization was not used.
    # Decision Tree splits data using threshold values, so different scales are not a problem.
    # Linear Regression can also work with features measured in different units.
    # However, without standardization, coefficient sizes should not be directly compared.


    # =========================
    # 6. Train/test split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\n===== TRAIN/TEST SPLIT =====")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # =========================
    # 7. Model 1: Linear Regression
    # =========================

    linear_model = LinearRegression()

    linear_model.fit(X_train, y_train)

    y_train_pred_lr = linear_model.predict(X_train)
    y_test_pred_lr = linear_model.predict(X_test)

    lr_train_r2 = r2_score(y_train, y_train_pred_lr)              # R^2 - coefficient of determination
    lr_test_r2 = r2_score(y_test, y_test_pred_lr)
    lr_mae = mean_absolute_error(y_test, y_test_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))

    print("\n===== LINEAR REGRESSION RESULTS =====")
    print("Train R2 score:", round(lr_train_r2, 4))
    print("Test R2 score:", round(lr_test_r2, 4))
    print("MAE:", round(lr_mae, 4))
    print("RMSE:", round(lr_rmse, 4))

    coefficients_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": linear_model.coef_
    })

    print("\nLinear Regression coefficients:")
    print(coefficients_df)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        y_test,
        y_test_pred_lr,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3
    )

    min_value = min(y_test.min(), y_test_pred_lr.min())
    max_value = max(y_test.max(), y_test_pred_lr.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linewidth=2,
        label="Perfect prediction"
    )

    plt.xlabel("Actual fuel consumption [l/100 km]")
    plt.ylabel("Predicted fuel consumption [l/100 km]")
    plt.title("Linear Regression: actual vs predicted fuel consumption")
    plt.legend()
    plt.tight_layout()

    linear_regression_chart_path = os.path.join(
        OUTPUT_DIR,
        "linear_regression_actual_vs_predicted.png"
    )

    plt.savefig(linear_regression_chart_path, dpi=200)
    plt.close()

    print("Saved Linear Regression chart:", linear_regression_chart_path)

    # =========================
    # 8. Model 2: Decision Tree Regressor
    # =========================

    tree_model = DecisionTreeRegressor(
        random_state=42
    )

    tree_model.fit(X_train, y_train)

    y_train_pred_tree = tree_model.predict(X_train)
    y_test_pred_tree = tree_model.predict(X_test)

    tree_train_r2 = r2_score(y_train, y_train_pred_tree)
    tree_test_r2 = r2_score(y_test, y_test_pred_tree)
    tree_mae = mean_absolute_error(y_test, y_test_pred_tree)
    tree_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred_tree)
    )

    print("\n===== DECISION TREE REGRESSOR RESULTS =====")
    print("Train R2 score:", round(tree_train_r2, 4))
    print("Test R2 score:", round(tree_test_r2, 4))
    print("MAE:", round(tree_mae, 4))
    print("RMSE:", round(tree_rmse, 4))

    # =========================
    # 9. Decision Tree hyperparameter optimization
    # =========================

    tree_parameters = {
        "max_depth": [2, 3, 4, 5, 6, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    tree_grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=tree_parameters,
        scoring="r2",
        cv=5,                      # folds
        n_jobs=-1                  # cores
    )

    tree_grid_search.fit(X_train, y_train)

    best_tree_model = tree_grid_search.best_estimator_       # best trained model

    y_train_pred_best_tree = best_tree_model.predict(X_train)
    y_test_pred_best_tree = best_tree_model.predict(X_test)

    best_tree_train_r2 = r2_score(
        y_train,
        y_train_pred_best_tree
    )

    best_tree_test_r2 = r2_score(
        y_test,
        y_test_pred_best_tree
    )

    best_tree_mae = mean_absolute_error(
        y_test,
        y_test_pred_best_tree
    )

    best_tree_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_test_pred_best_tree
        )
    )

    print("\n===== OPTIMIZED DECISION TREE RESULTS =====")
    print("Best parameters:", tree_grid_search.best_params_)
    print(
        "Best cross-validation R2:",
        round(tree_grid_search.best_score_, 4)
    )
    print("Train R2 score:", round(best_tree_train_r2, 4))
    print("Test R2 score:", round(best_tree_test_r2, 4))
    print("MAE:", round(best_tree_mae, 4))
    print("RMSE:", round(best_tree_rmse, 4))

    # =========================
    # Optimized Decision Tree: actual vs predicted chart
    # =========================

    plt.figure(figsize=(7, 5))

    plt.scatter(
        y_test,
        y_test_pred_best_tree,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3
    )

    min_value_tree = min(
        y_test.min(),
        y_test_pred_best_tree.min()
    )

    max_value_tree = max(
        y_test.max(),
        y_test_pred_best_tree.max()
    )

    plt.plot(
        [min_value_tree, max_value_tree],
        [min_value_tree, max_value_tree],
        linewidth=2,
        label="Perfect prediction"
    )

    plt.xlabel("Actual fuel consumption [l/100 km]")
    plt.ylabel("Predicted fuel consumption [l/100 km]")
    plt.title(
        "Optimized Decision Tree: actual vs predicted fuel consumption"
    )

    plt.legend()
    plt.tight_layout()

    optimized_tree_chart_path = os.path.join(
        OUTPUT_DIR,
        "optimized_tree_actual_vs_predicted.png"
    )

    plt.savefig(
        optimized_tree_chart_path,
        dpi=200
    )

    plt.close()

    print(
        "Saved Optimized Decision Tree chart:",
        optimized_tree_chart_path
    )

    # =========================
    # 10. Model comparison
    # =========================

    comparison_df = pd.DataFrame({
        "model": [
            "Linear Regression",
            "Decision Tree",
            "Optimized Decision Tree"
        ],
        "train_r2": [
            lr_train_r2,
            tree_train_r2,
            best_tree_train_r2
        ],
        "test_r2": [
            lr_test_r2,
            tree_test_r2,
            best_tree_test_r2
        ],
        "mae": [
            lr_mae,
            tree_mae,
            best_tree_mae
        ],
        "rmse": [
            lr_rmse,
            tree_rmse,
            best_tree_rmse
        ]
    })

    print("\n===== MODEL COMPARISON =====")
    print(comparison_df.round(4))

    comparison_csv_path = os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_csv_path,
        index=False
    )

    print("Saved model comparison table:", comparison_csv_path)

    # R2 comparison chart
    comparison_df.plot(
        x="model",
        y=["train_r2", "test_r2"],
        kind="bar",
        figsize=(9, 5),
        rot=0                     #rotation
    )

    plt.ylabel("R2 score")
    plt.xlabel("Model")
    plt.title("Train and test R2 comparison")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    r2_chart_path = os.path.join(
        OUTPUT_DIR,
        "model_r2_comparison.png"
    )

    plt.savefig(r2_chart_path, dpi=200)
    plt.close()

    print("Saved R2 comparison chart:", r2_chart_path)

    # MAE and RMSE comparison chart
    comparison_df.plot(
        x="model",
        y=["mae", "rmse"],
        kind="bar",
        figsize=(9, 5),
        rot=0
    )

    plt.ylabel("Error [l/100 km]")
    plt.xlabel("Model")
    plt.title("MAE and RMSE comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    error_chart_path = os.path.join(
        OUTPUT_DIR,
        "model_error_comparison.png"
    )

    plt.savefig(error_chart_path, dpi=200)
    plt.close()

    print("Saved error comparison chart:", error_chart_path)

    best_model_index = comparison_df["test_r2"].idxmax()
    best_model_name = comparison_df.loc[
        best_model_index,
        "model"
    ]

    print("\nBest model based on Test R2:", best_model_name)
