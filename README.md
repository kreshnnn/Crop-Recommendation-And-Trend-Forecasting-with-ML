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

## Applications

- The model provides crop recommendations based on various environmental and soil factors such as state, district, and season.
- It supports farmers in making better-informed decisions regarding crop selection, ultimately boosting productivity.
- The system optimizes resource allocation by suggesting the most suitable crops, promoting more sustainable farming practices.

## Advantages

- The model enhances farming efficiency by recommending the best crops for the given conditions, leading to improved yields.
- It is scalable, allowing for expansion to other regions with minimal adjustments as new data becomes available.
- It encourages sustainability by aligning crop recommendations with available resources, reducing waste and overuse of resources.

## Limitations

- The performance of the model is highly reliant on the quality and accuracy of the data, which may not always be complete or consistent.
- The model does not factor in unexpected external influences such as extreme weather events, pest infestations, or disease outbreaks.
- The focus is primarily on providing short-term recommendations, which may not account for long-term agricultural trends or challenges.

## Future Scope

- Integrating real-time data from weather stations and soil sensors could enhance the model's accuracy and provide more dynamic crop recommendations.
- Future updates could include the addition of features to predict pest infestations or disease outbreaks, helping farmers mitigate risks.
- Developing a mobile application would allow farmers to access crop recommendations and forecasts on-the-go, making the system more accessible and user-friendly.

## Contributions

Contributions are welcome! Feel free to fork the repository, create issues, and submit pull requests to improve the functionality, add new features, or enhance the model.

## Project Status

Completed — The core functionality of crop recommendation has been implemented. Further enhancements, including additional features and advanced forecasting, are encouraged.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Thank You! 
