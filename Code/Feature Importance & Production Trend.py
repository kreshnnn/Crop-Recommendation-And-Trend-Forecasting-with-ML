import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
file_path = '/Users/kreshngautam/Downloads/Personal Projects/Crop Production in India/Crop Production data.csv'
df = pd.read_csv(file_path, dtype={'State_Name': 'str', 'District_Name': 'str', 'Season': 'str', 'Crop': 'str'}, low_memory=False)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42)

# Step 5: Initialize and Train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x--------x--------x-------x-------x-------x--------x-------x-------x-------x-------x

# IMPORTING FEATURE IMPORTANCE DATA FROM ML MODEL

# Getting the data from model
importance = rf_model.feature_importances_

# Display feature importance
for i, v in enumerate(importance):
    print(f'Feature: {X.columns[i]}, Importance: {v:.5f}')

# Visualize Feature Importance with a limited x-axis range
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=X.columns)
plt.title('Feature Importance for Crop Production Prediction')

# Set x-axis limits from 0 to 1 with ticks at intervals of 0.1
plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.1))  # Ticks from 0 to 1 at intervals of 0.1

plt.xlabel('Importance')
plt.ylabel('Features')

plt.tight_layout()
plt.show()

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x--------x--------x-------x-------x-------x--------x-------x-------x-------x-------x

# GRAPH OF ACTUAL VS PREDICTED VALUE OF PRODUCTION TREND- focusing on year-by-year performance on the basis of individual data points

# Compare actual vs predicted values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
comparison['Crop_Year'] = X_test['Crop_Year'].values  # Add Crop_Year for the x-axis
print(comparison.head())

# Sort the comparison DataFrame by Crop Year for cleaner lines
comparison = comparison.sort_values(by='Crop_Year')

# Plot Actual vs Predicted Values
plt.figure(figsize=(12, 6))

# Plot actual values
plt.plot(comparison['Crop_Year'], comparison['Actual'], label='Actual Production', marker='o', linestyle='-', color='blue')

# Plot predicted values
plt.plot(comparison['Crop_Year'], comparison['Predicted'], label='Predicted Production', marker='o', linestyle='--', color='red')

# Add titles and labels
plt.title('Actual vs Predicted Crop Production', fontsize=16)
plt.xlabel('Crop Year', fontsize=12)
plt.ylabel('Production (in Metric Tonnes)', fontsize=12)

# Add a legend & displaing the plot
plt.legend()
plt.tight_layout()
plt.show()

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x--------x--------x-------x-------x-------x--------x-------x-------x-------x-------x

# GRAPH OF AVERAGE PRODUCTION TREND- average values (aggregated by year), focusing on the general/overall long-term trends of production rather than individual data points

# Aggregate data by Crop Year (mean production per year)
comparison_agg = comparison.groupby('Crop_Year').mean().reset_index()

# Plot Aggregated Actual vs Predicted Values
plt.figure(figsize=(12, 6))

# Plot actual values (aggregated)
plt.plot(comparison_agg['Crop_Year'], comparison_agg['Actual'], label='Actual Production', marker='o', linestyle='-', color='blue')

# Plot predicted values (aggregated)
plt.plot(comparison_agg['Crop_Year'], comparison_agg['Predicted'], label='Predicted Production', marker='o', linestyle='--', color='red')

# Add titles and labels
plt.title('Aggregated Actual vs Predicted Crop Production', fontsize=16)
plt.xlabel('Crop Year', fontsize=12)
plt.ylabel('Production (in Metric Tonnes)', fontsize=12)

# Add a legend & display the plot
plt.legend()
plt.tight_layout()
plt.show()

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

# Accuracy of model
accuracy = 100 - median_error_percentage

# Display the capped error percentages and the overall mean and median
print(f'Capped Error Percentage for first few predictions: \n{error_percentage_capped[:5]}')
print(f'Mean Error Percentage: {mean_error_percentage:.2f}%')
print(f'Median Error Percentage: {median_error_percentage:.2f}%')
print(f'The Accuracy of the Model is: {accuracy:.2f}%')

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x--------x--------x-------x-------x-------x--------x-------x-------x-------x-------x