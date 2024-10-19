# Municipal Waste Management Cost Prediction

This project focuses on predicting the per capita municipal waste management costs using machine learning models applied to a dataset that includes socio-economic, geographical, and waste management-specific attributes from multiple municipalities. By leveraging advanced feature engineering and regression techniques, we aim to create a robust predictive model that can assist municipalities in managing their waste efficiently and economically.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Project Pipeline](#project-pipeline)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Approach](#modeling-approach)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Problem Statement

The project addresses the problem of predicting the cost per capita for municipal waste management across different regions using machine learning. Accurately predicting these costs enables municipalities to budget more effectively and optimize waste disposal operations, with the potential to significantly reduce overhead while maintaining or improving service quality.

## Dataset Overview

The dataset consists of various features that describe demographic, economic, and geographic properties of the municipalities, as well as details on waste management practices such as waste types and waste disposal methods. The main target variable is the **cost per capita (EUR)** for managing municipal waste.

The dataset includes:

- 10,000+ municipalities
- 30+ features
- Multi-class and continuous variables

## Project Pipeline

The workflow for the project is divided into the following steps:

1. **Data Collection and Exploration**: Loading and understanding the data, detecting outliers, and identifying missing values.
2. **Data Preprocessing**: Handling missing data, transforming categorical variables, and feature scaling.
3. **Feature Engineering**: Creating new features such as waste per person, log transformations, interaction terms, and polynomial features.
4. **Model Training**: Training multiple regression models to predict the cost per capita.
5. **Hyperparameter Tuning**: Fine-tuning model parameters using cross-validation and grid search.
6. **Evaluation**: Evaluating the model using metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R² score.
7. **Model Interpretation**: Analyzing feature importance using SHAP values and feature coefficients.

## Features

The following features were considered for model development:

| Variable        | Description                                              |
|-----------------|----------------------------------------------------------|
| `region`        | Geographic region (South, Center, North)                 |
| `province`      | Province of the municipality                              |
| `pop`           | Population size of the municipality                       |
| `pden`          | Population density (people per km²)                       |
| `wden`          | Waste per km²                                             |
| `urb`           | Urbanization index (1 low, 3 high)                        |
| `organic`       | Percentage of organic waste                               |
| `plastic`       | Percentage of plastic waste                               |
| `gdp`           | Municipal GDP (log)                                       |
| `wage`          | Taxable income EUR (log)                                  |
| `s_landfill`    | Share of waste sent to landfill                           |
| `msw`           | Municipal solid waste (kg)                                |
| `roads`         | Km of roads within the municipality                       |
| `finance`       | Municipal revenues EUR (log)                              |
| ...             | (Full feature list in `features.csv`)                     |

## Data Preprocessing

To ensure the dataset is ready for modeling, the following preprocessing steps were applied:

1. **Missing Value Imputation**:
   - Missing values were imputed using a combination of mean imputation for numerical features and mode imputation for categorical features.

2. **Categorical Encoding**:
   - Categorical variables such as `region`, `province`, and `fee_scheme` were transformed using **one-hot encoding**.
   - Binary variables like `isle` and `sea` were retained as dummy variables.

3. **Feature Scaling**:
   - Continuous variables like `gdp`, `wage`, and `population` were scaled using **MinMaxScaler** to normalize the range of values for improved model performance.

4. **Log Transformation**:
   - Log transformation was applied to skewed features like `gdp`, `population`, and `taxable_income` to reduce the effect of outliers and better approximate a normal distribution.

## Modeling Approach

Several machine learning algorithms were implemented to predict the target variable `cost per capita (EUR)`:

1. **Linear Regression**: Baseline model to establish performance benchmarks.
2. **Ridge Regression**: To address multicollinearity between features and enhance model generalization.
3. **Random Forest Regressor**: An ensemble learning method, particularly useful for capturing non-linear relationships in the dataset.
4. **XGBoost**: An optimized gradient boosting framework that builds trees sequentially to minimize prediction error.
5. **LightGBM**: Gradient boosting model specifically designed for speed and efficiency on large datasets.

### Hyperparameter Tuning

For each model, hyperparameter optimization was performed using **GridSearchCV** and **RandomizedSearchCV** with 5-fold cross-validation. The following hyperparameters were tuned:

- `alpha` (Ridge Regression)
- `n_estimators`, `max_depth`, `min_samples_split` (Random Forest)
- `learning_rate`, `n_estimators`, `max_depth` (XGBoost and LightGBM)

## Evaluation Metrics

The models were evaluated using the following performance metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions.
- **Root Mean Square Error (RMSE)**: Penalizes larger errors more than smaller errors, making it useful for catching outlier predictions.
- **R² Score**: Represents the proportion of variance in the dependent variable that can be explained by the independent variables.

## Installation

To run this project, clone the repository and install the required dependencies:

```
git clone https://github.com/your-username/waste-management-cost-prediction.git
cd waste-management-cost-prediction
pip install -r requirements.txt
```
## Usage
To preprocess the dataset and train the model, follow these steps:
```
# Preprocess the dataset
python preprocess_data.py --input data/municipal_waste.csv --output data/cleaned_data.csv

# Train the machine learning model
python train_model.py --dataset data/cleaned_data.csv --model xgboost
```
Model predictions and evaluation results will be stored in the output/ folder.

## Model Performance
The final model's performance on the test set is as follows:

Mean Absolute Error (MAE): 25.34 EUR
Root Mean Square Error (RMSE): 35.21 EUR
R² Score: 0.78
Feature importance analysis using SHAP reveals that population density, GDP, waste type proportions, and urbanization index are significant predictors of waste management costs.

## Contributing
We welcome contributions from the community. To contribute:
1. Fork the repository.
2. Create a new branch with your feature/bugfix.
3. Submit a pull request for review.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
