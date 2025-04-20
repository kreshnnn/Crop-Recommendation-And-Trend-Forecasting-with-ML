import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Loading the dataset
file_path = '/Users/kreshngautam/Downloads/Personal Projects/Crop Production in India/Crop Production data.csv'
df = pd.read_csv(file_path)

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x--------x--------x-------x-------x-------x--------x-------x-------x-------x-------x

# DATA PROCESSING & MODELLING FOR CROP DATA PREDICTION VIA RANDOM FOREST GENERATOR MODEL

# Check for missing values in the dataset and drop rows with missing production values
df = df.dropna(subset=['Production'])

# Encode categorical variables like State_Name, District_Name, Season, and Crop using LabelEncoder
label_encoder = LabelEncoder()
df['State_Name'] = label_encoder.fit_transform(df['State_Name'])
df['District_Name'] = label_encoder.fit_transform(df['District_Name'])
df['Season'] = label_encoder.fit_transform(df['Season'])
df['Crop'] = label_encoder.fit_transform(df['Crop'])

# Step 3: Define Features (X) and Target (y)
X = df[['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area']]
y = df['Production']

# Step 4: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and Train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = rf_model.predict(X_test)

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x--------x--------x-------x-------x-------x--------x-------x-------x-------x-------x

# ERROR PERCENTAGE & ACCURACY CALCULATION

# Ensure there are no zero actual values to avoid division by zero errors
non_zero_actuals = y_test != 0  # Mask to ignore zero actuals

# Filter actual and predicted values where actual is non-zero
filtered_y_test = y_test[non_zero_actuals]
filtered_y_pred = y_pred[non_zero_actuals]

# Calculate the absolute percentage error for each prediction
error_percentage = np.abs((filtered_y_test - filtered_y_pred) / filtered_y_test) * 100

# Cap the error percentage at 100% to avoid skewing by large outliers
error_percentage_capped = np.clip(error_percentage, 0, 100)

# Calculate the mean and median error percentage across all predictions
mean_error_percentage = np.mean(error_percentage_capped)
median_error_percentage = np.median(error_percentage_capped)

# R² Score
r2 = r2_score(y_test, y_pred)

# Accuracy of model
accuracy = 100 - median_error_percentage

# Display the capped error percentages and the overall mean and median
print(f'Capped Error Percentage for first few predictions: \n{error_percentage_capped[:5]}')
print(f'Mean Error Percentage: {mean_error_percentage:.2f}%')
print(f'Median Error Percentage: {median_error_percentage:.2f}%')
print(f'The Accuracy of the Model is: {accuracy:.2f}%')
print(f"R² Score: {r2:.2f}")

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x--------x--------x-------x-------x-------x--------x-------x-------x-------x-------x

# FUTURE CROP YIELD PREDICTION MODEL

# Future data input
future_data = pd.DataFrame({
    'State_Name': [0],  # Replace 0 with the appropriate state encoding
    'District_Name': [0],  # Replace 0 with the appropriate district encoding
    'Crop_Year': [2025],  # Future year for which you want to predict
    'Season': [0],  # Replace with the appropriate season encoding
    'Crop': [0],  # Replace with the appropriate crop encoding
    'Area': [10000]  # Replace with the appropriate area in KM
})

# Make predictions using the previously trained Random Forest model
future_prediction = rf_model.predict(future_data)

# Output the predicted production for 2025
print(f'Predicted Production for the Year 2025: {future_prediction[0]:.2f} Metric Tonnes')

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x--------x--------x-------x-------x-------x--------x-------x-------x-------x-------x