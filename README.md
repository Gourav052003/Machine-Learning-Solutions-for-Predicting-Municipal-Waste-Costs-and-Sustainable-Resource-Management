# Municipal Waste Management Cost Prediction

This project aims to predict the per capita cost of municipal waste management using a comprehensive dataset covering various socio-economic, geographic, and environmental factors across multiple regions and municipalities. The prediction model helps municipalities optimize waste management strategies and reduce costs.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Business Value](#business-value)
- [Machine Learning Approach](#machine-learning-approach)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Municipal waste management is a critical component of public services, affecting environmental sustainability and municipal budgets. The project leverages machine learning to predict the cost per capita (in EUR) for waste management in municipalities based on several socio-economic, geographic, and waste-related features.

## Dataset

The dataset used in this project includes detailed data points from municipalities across different regions. It covers variables such as population density, waste types (organic, plastic, metal, etc.), urbanization index, altitude, GDP, and more. 

The main target variable for prediction is the **cost per capita (EUR)** for waste management. 

For a detailed description of the dataset, refer to the [dataset section](#features) below.

## Business Value

Efficiently managing waste is crucial for reducing environmental impact and optimizing municipal costs. This model:
- Helps municipalities predict future costs and budget accordingly.
- Identifies key cost drivers, allowing targeted improvements in waste management.
- Supports the development of cost-effective waste disposal strategies, like Pay-As-You-Throw (PAYT).

## Machine Learning Approach

This project uses various regression models to predict waste management costs, including:
- **Linear Regression**
- **Random Forest**
- **XGBoost**

After evaluating model performance using metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE), the best-performing model is selected for the final predictions.

## Features

The dataset includes the following key features:

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

## Installation

To run this project, clone the repository and install the required dependencies:

```
git clone https://github.com/your-username/waste-management-cost-prediction.git
cd waste-management-cost-prediction
pip install -r requirements.txt
```
