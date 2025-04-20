import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Loading the dataset
file_path = '/Users/kreshngautam/Downloads/Personal Projects/Crop Production in India/Crop Production data.csv'
df = pd.read_csv(file_path)

# Check the first few rows of the dataset
df.head()

# General statistics
print(df.describe())

# Check for any of the remaining missing values
print(df.isnull().sum())

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x

# YEARLY TREND ANALYSIS OF OVERALL PRODUCTION

# Group the data by Crop_Year and calculate total production per year
yearly_production = df.groupby('Crop_Year')['Production'].sum().reset_index()

# Plot crop production over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_production, x='Crop_Year', y='Production', marker='o')
plt.title('Total Crop Production Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.grid(True)
plt.show()

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x

# Group the data by State and calculate total production for each state (in kgs)
state_production = df.groupby('State_Name')['Production'].sum().reset_index()

# Convert production from kilograms to metric tonnes
state_production['Production'] = state_production['Production'] / 1000  # Convert to metric tonnes

# Sort and plot production by state
state_production = state_production.sort_values(by='Production', ascending=False)

plt.figure(figsize=(14, 7))  # Adjust the figure size for better label visibility
sns.barplot(data=state_production, x='State_Name', y='Production')

# Set title and labels with units
plt.title('Total Crop Production by State (in Metric Tonnes)', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Total Production (in metric tonnes)', fontsize=12)

# Rotate x-axis labels and set limits on y-axis
plt.xticks(rotation=90, ha='center', fontsize=10)
plt.ylim(0, state_production['Production'].max() * 1.1)  # Adjust y-axis range to give some padding

# Format y-axis in metric tonnes
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Ensure everything fits within the figure
plt.tight_layout()  
plt.show()

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x

# Group by Crop and calculate total production for each crop type (in kgs)
crop_production = df.groupby('Crop')['Production'].sum().reset_index()

# Convert production from kilograms to metric tonnes
crop_production['Production'] = crop_production['Production'] / 1000  # Convert to metric tonnes

# Sort and visualize top 10 crops by production
crop_production = crop_production.sort_values(by='Production', ascending=False).head(10)  # Top 10 crops

plt.figure(figsize=(14, 7))  # Adjust the figure size for better label visibility
sns.barplot(data=crop_production, x='Crop', y='Production')

# Set title and labels with units
plt.title('Top 10 Crop Productions (in Metric Tonnes)', fontsize=16)
plt.xlabel('Crop', fontsize=12)
plt.ylabel('Total Production (in metric tonnes)', fontsize=12)

# Rotate x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right', fontsize=10)

# Set appropriate limits on y-axis to ensure visibility of all values
plt.ylim(0, crop_production['Production'].max() * 1.1)  # Add 10% padding above the max value

# Format y-axis for better readability (with commas for large numbers)
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()  # Ensure the layout fits well
plt.show()

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x

# Correlation between numerical columns
corr_matrix = df[['Area', 'Production']].corr()

# Visualize the correlation matrix with adjusted figure size and proper labels
plt.figure(figsize=(8, 6))  # Adjust the figure size for better clarity
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12})

# Set a title
plt.title('Correlation Matrix', fontsize=14)

# Adjusting the ticks on both axes
plt.xticks(ticks=[0.5, 1.5], labels=['Area', 'Production'], rotation=0, fontsize=12)
plt.yticks(ticks=[0.5, 1.5], labels=['Area', 'Production'], rotation=0, fontsize=12)

plt.tight_layout()  # Ensure that everything fits well within the plot
plt.show()

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x

# Group data by Season and Crop, and calculate total production for each
season_crop_top = df.groupby(['Season', 'Crop'])['Production'].sum().reset_index()

# Convert production from kilograms to metric tonnes
season_crop_top['Production'] = season_crop_top['Production'] / 1000  # Convert to metric tonnes

# Sort data by Production within each Season
season_crop_top = season_crop_top.groupby('Season').apply(lambda x: x.sort_values(by='Production', ascending=False)).reset_index(drop=True)

# Plot the top crops grown in each season
plt.figure(figsize=(14, 7))

# Plot the data with production values in metric tonnes
sns.barplot(data=season_crop_top, x='Season', y='Production', hue='Crop', dodge=False)

# Set title and labels
plt.title('Top Crops in Each Season by Production (in Metric Tonnes)', fontsize=16)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Total Production (in Metric Tonnes)', fontsize=12)

# Ensure legend placement
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), title='Crop')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)

# Adjust y-axis to handle large range of production values
plt.ylim(0, season_crop_top['Production'].max() * 1.1)  # Add 10% padding to the maximum value for better visualization

plt.tight_layout()  # Ensure the plot fits well
plt.show()

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x

# OUTLINERS

# Plot the boxplot for Production to detect outliers
plt.figure(figsize=(10, 6))

# Create a boxplot to visualize outliers in the Production column
sns.boxplot(x=df['Production'])

# Add title and labels for better clarity
plt.title('Outlier Detection in Production', fontsize=16)
plt.xlabel('Production (in metric tonnes)', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()

#--------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x
