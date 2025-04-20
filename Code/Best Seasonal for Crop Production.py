import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
crop_data_df = pd.read_csv('/Users/kreshngautam/Documents/All Projects/Personal Projects/Python Projects/Crop Production in India/Crop Production data.csv')

# Fill missing values if necessary (optional, depending on your data)
crop_data_df = crop_data_df.dropna()

# Encode the categorical variables using LabelEncoder
label_encoder_state = LabelEncoder()
label_encoder_district = LabelEncoder()
label_encoder_crop = LabelEncoder()
label_encoder_season = LabelEncoder()

crop_data_df['State_Code'] = label_encoder_state.fit_transform(crop_data_df['State_Name'])
crop_data_df['District_Code'] = label_encoder_district.fit_transform(crop_data_df['District_Name'])
crop_data_df['Crop_Code'] = label_encoder_crop.fit_transform(crop_data_df['Crop'])
crop_data_df['Season_Code'] = label_encoder_season.fit_transform(crop_data_df['Season'])

# Select relevant features (State, District, Crop, and Area) and target (Season)
X = crop_data_df[['State_Code', 'District_Code', 'Crop_Code', 'Area']]
y = crop_data_df['Season_Code']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification report to analyze performance
print(classification_report(y_test, y_pred))

# Predict season for new inputs (State, District, Crop, Area)
new_data = [[0, 0, 0, 1000]]  # Example: State 0, District 0, Crop 0, Area 1000
predicted_season = rf_classifier.predict(new_data)

# Decode the predicted season to its original name
predicted_season_name = label_encoder_season.inverse_transform(predicted_season)
print(f"Predicted Optimal Season: {predicted_season_name[0]}")
