# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('medicines.csv')
data= data[['Drug Name', 'Drug Dosage Expiry Date']]

# Preprocess the dataset
dates = pd.to_datetime(data['Drug Dosage Expiry Date'])
data['expiration_month'] = dates.dt.month
data['expiration_year'] = dates.dt.year
data.shape
data.head()

# REMOVING NON-USEFUL DATA
data = data[data['expiration_year'] == 2018]
data.shape
data.head()

# CHECKING FOR ANY NULL VALUES 
data['Drug Dosage Expiry Date'].isna().sum()
data['expiration_month'].isna().sum()
data['expiration_year'].isna().sum()

# CHECKING UNIQUE VALUES FOR MONTH AND YEAR
unique_years = data['expiration_year'].unique()

# Print the unique values
print(unique_years)

unique_month = data['expiration_month'].unique()

# Print the unique values
print(unique_month)
# Get the unique values in the expiration_year column in descending order
unique_month_desc = data['expiration_month'].sort_values(ascending=False).unique()

# Print the unique values in descending order
print(unique_month_desc)
