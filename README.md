# Crop Recommendation and Agricultural Trend Forecasting with Machine Learning

## Overview

This project uses machine learning techniques to recommend suitable crops based on environmental and soil factors like state, district, crop, area, and season. The model analyzes these features to suggest optimal crops for specific regions, improving farming efficiency and productivity.

Additionally, the project forecasts agricultural trends using historical data, helping stakeholders make informed decisions on crop planning, resource allocation, and market strategies.

Built with **Pandas**, **Numpy**, **Scikit-learn**, **Matplotlib**, and **Seaborn**, this project merges traditional agricultural insights with modern machine learning techniques to optimize crop selection and forecast agricultural trends.

## Table of Contents

1. [Built With](#built-with)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Model Training](#model-training)
5. [Usage](#usage)
6. [Applications](#applications)
7. [Advantages](#advantages)
8. [Limitations](#limitations)
9. [Future Scope](#future-scope)
10. [Contributions](#contributions)
11. [Project Status](#project-status)
12. [License](#license)

## Built With

- **Pandas**: Data manipulation and analysis
- **Numpy**: Numerical computations
- **Scikit-learn**: Machine learning models (Random Forest, etc.)
- **Matplotlib**: Data visualization
- **Seaborn**: Advanced visualization for statistical graphics

## Dataset

The dataset used in this project includes the following factors:
- **State**: Geographic location of the agricultural region
- **District**: Local level data within the state
- **Crop**: The crop being analyzed
- **Area**: The size of land used for cultivation
- **Season**: The season in which the crop is cultivated

These features are used to recommend the best crops for a specific region based on historical agricultural data.

## Methodology

1. **Data Preprocessing**:
   - Cleaned the data by handling missing values and scaling numerical features.
   - Converted categorical features (state, district, season) into numerical formats using encoding techniques.

2. **Feature Engineering**:
   - Selected key features (state, district, crop, area, season) based on their importance for crop recommendation.
   - Dropped irrelevant or redundant features.

3. **Model Selection**:
   - Trained a **Random Forest Classifier** to predict crop recommendations based on the input features.
   - Used **train-test split** to evaluate model performance.

## Model Training

The model was trained using the **Random Forest Classifier** from `scikit-learn`:
- Split the dataset into training and testing sets using **train_test_split**.
- Evaluated the model using accuracy, precision, and recall metrics to ensure reliable predictions for crop recommendations.

## Usage

### Requirements

- Python 3.x
- **Pandas**, **Numpy**, **Scikit-learn**, **Matplotlib**, **Seaborn** (installed via `pip install -r requirements.txt`)

### Running the Model

1. Clone this repository:
   ```bash
   git clone https://github.com/kreshnnn/Crop-Recommendation-and-Trend-Forecasting-with-ML.git
