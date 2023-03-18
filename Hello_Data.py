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


# APPLYING ONE HOT ENCODING
# Use the pandas get_dummies() function to one-hot encode the "expiration_month" column
one_hot = pd.get_dummies(data['expiration_month'])

# Rename the columns to be more descriptive
one_hot.columns = ['month_'+str(i) for i in range(1,13)]

# Concatenate the one-hot encoded data back onto the original dataframe
data = pd.concat([data, one_hot], axis=1)

data.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data['Drug Name'])

# TRANSFORMING THE DATASET
data['Drug_Id'] = le.transform(data['Drug Name'])
data.head()
data = data.drop('Drug Name', axis=1)
data = data.drop('Drug Dosage Expiry Date', axis=1)
data

# APPLYING CLUSTERING
kmeans = KMeans(n_clusters=12)
kmeans.fit(data)
labels = kmeans.labels_
labels

# PLOTTING 
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data['expiration_month'], data['expiration_year'], c=labels, cmap='rainbow')
ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_title('Month-wise Clustering of Medicines by Expiration Date')
ax.set_ylim(2017, 2019) # set y-axis limits
plt.show()

data = data.reset_index(drop=True)

# RELABELLING THE PLOT

# define colors for each cluster
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan', 'magenta']

# assign colors to each point based on cluster labels
point_colors = [colors[label] for label in labels]

# plot the clusters with assigned colors
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data['expiration_month'], data['expiration_year'], c=point_colors)
ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_title('Month-wise Clustering of Medicines by Expiration Date')
plt.show()

count = data['Drug_Id'].nunique()

print(count)
data

# Group the medicines based on expiration month
groups = data.groupby('expiration_month')

# Assign a group number to each group
group_num = 0
for name, group in groups:
    data.loc[group.index, 'group'] = group_num
    group_num += 1

data

# REPLOTTING THE PLOT

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data['expiration_month'], data['group'], c=data['group'], cmap='rainbow')
ax.set_xlabel('Expiration Month')
ax.set_ylabel('Group')
ax.set_title('Grouped Medicines by Expiration Month')
plt.show()

# GETTING EACH  CLUSTER VALUES

group_1_drugs = data[data['group'] == 1]['Drug_Id'].tolist()
print(group_1_drugs)

# EVALUATION

from sklearn.metrics import silhouette_score
# calculate the silhouette score for the model
silhouette_avg = silhouette_score(data, kmeans.labels_)

print("The average silhouette score is :", silhouette_avg)

from sklearn.metrics import calinski_harabasz_score
score = calinski_harabasz_score(data, kmeans.labels_)
print(score)
