import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the original dataset (replace with your file path)
original_file_path = '/Users/kreshngautam/Downloads/Personal Projects/Crop Production in India/Crop Production data.csv'
df_original = pd.read_csv(original_file_path)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Create the mapping dictionaries for each categorical column
state_name_mapping = {i: v for i, v in enumerate(label_encoder.fit(df_original['State_Name']).classes_)}
district_name_mapping = {i: v for i, v in enumerate(label_encoder.fit(df_original['District_Name']).classes_)}
season_mapping = {i: v for i, v in enumerate(label_encoder.fit(df_original['Season']).classes_)}
crop_mapping = {i: v for i, v in enumerate(label_encoder.fit(df_original['Crop']).classes_)}

# Convert mappings to DataFrames
state_mapping_df = pd.DataFrame(list(state_name_mapping.items()), columns=['Encoded State_Name', 'Original State_Name'])
district_mapping_df = pd.DataFrame(list(district_name_mapping.items()), columns=['Encoded District_Name', 'Original District_Name'])
season_mapping_df = pd.DataFrame(list(season_mapping.items()), columns=['Encoded Season', 'Original Season'])
crop_mapping_df = pd.DataFrame(list(crop_mapping.items()), columns=['Encoded Crop', 'Original Crop'])

# Combine all mappings into a single DataFrame by aligning columns and filling shorter columns with NaN
combined_df = pd.concat([state_mapping_df, district_mapping_df, season_mapping_df, crop_mapping_df], axis=1)

# Save the combined mapping to a CSV file
combined_df.to_csv('Combined_Categorical_Mappings.csv', index=False)

print("Combined mapping saved as 'Combined_Categorical_Mappings.csv'")
