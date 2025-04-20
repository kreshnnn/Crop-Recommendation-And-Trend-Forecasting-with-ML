import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
crop_data_df = pd.read_csv('/Users/kreshngautam/Downloads/Personal Projects/Crop Production in India/Crop Production data.csv')

# Fill missing values if necessary
crop_data_df = crop_data_df.dropna()

# Encode the categorical variables using LabelEncoder
label_encoder_state = LabelEncoder()
label_encoder_district = LabelEncoder()
label_encoder_season = LabelEncoder()
label_encoder_crop = LabelEncoder()

crop_data_df['State_Code'] = label_encoder_state.fit_transform(crop_data_df['State_Name'])
crop_data_df['District_Code'] = label_encoder_district.fit_transform(crop_data_df['District_Name'])
crop_data_df['Season_Code'] = label_encoder_season.fit_transform(crop_data_df['Season'])
crop_data_df['Crop_Code'] = label_encoder_crop.fit_transform(crop_data_df['Crop'])

# Log-transform the 'Area' feature
crop_data_df['Log_Area'] = np.log1p(crop_data_df['Area'])

# Create a binary label for crop suitability (based on a threshold of production levels)
crop_data_df['Suitability'] = np.where(crop_data_df['Production'] > crop_data_df['Production'].median(), 1, 0)

# Select relevant features (State, District, Season, Area) and target (Suitability)
X = crop_data_df[['State_Code', 'District_Code', 'Season_Code', 'Log_Area']]
y = crop_data_df['Suitability']

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
print(classification_report(y_test, y_pred))

# Predict crop suitability for new inputs (State, District, Season, Log_Area)
new_data = [[0, 0, 0, np.log1p(1000)]]  
predicted_suitability = rf_classifier.predict(new_data)
print(f"Predicted Suitability: {'Suitable' if predicted_suitability[0] == 1 else 'Not Suitable'}")
