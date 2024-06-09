#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


df= pd.read_csv('NHANES_age_prediction 2.csv')
df.head(100)


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.info


# In[6]:


df.info()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('NHANES_age_prediction 2.csv')

# Create subplots for each of the headers
headers = [
    'age_group',
    'RIAGENDR_GENDER',
    'BMXBMI_BMI',
    'LBXGLU_GLUCOSE',
    'DIQ010_DIBETIC',
    'LBXGLT_ORAL',
    'LBXIN_INSULIN'
]

# Create subplots with 2 rows and 3 columns (for the 6 headers)
fig, axes = plt.subplots(3, 3, figsize=(17, 10))

# Set the X axis as RIDAGEYR_AGE for all plots
x = data['RIDAGEYR_AGE']

# Plot each header on a separate subplot
for i, header in enumerate(headers):
    row = i // 3
    col = i % 3
    ax = axes[row, col]

    y = data[header]

    ax.scatter(x, y, alpha=0.5)
    ax.set_xlabel('RIDAGEYR_AGE')
    ax.set_ylabel(header)

    ax.set_title(f"Plot of {header}")

# Adjust the layout and display the plots
plt.tight_layout()
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the csv file
data = pd.read_csv('NHANES_age_prediction 2.csv')

# Set the Seaborn style
sns.set(style="whitegrid")

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data with vibrant colors
sns.lineplot(x='RIDAGEYR_AGE', y='BMXBMI_BMI', data=data, label='BMXBMI_BMI')
sns.lineplot(x='RIDAGEYR_AGE', y='LBXGLU_GLUCOSE', data=data, label='LBXGLU_GLUCOSE')
sns.lineplot(x='RIDAGEYR_AGE', y='DIQ010_DIBETIC', data=data, label='DIQ010_DIBETIC')
sns.lineplot(x='RIDAGEYR_AGE', y='LBXGLT_ORAL', data=data, label='LBXGLT_ORAL')
sns.lineplot(x='RIDAGEYR_AGE', y='LBXIN_INSULIN', data=data, label='LBXIN_INSULIN')

# Set the labels and title
ax.set_xlabel('Age (Years)')
ax.set_ylabel('Values')
ax.set_title('NHANES Age Predictions')

# Display the legend with a title
ax.legend(title='Variables')

# Customize the grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Read the csv file
data = pd.read_csv('NHANES_age_prediction 2.csv')

# Encode the "age_group" column
le = LabelEncoder()
data['age_group'] = le.fit_transform(data['age_group'])

# Select the relevant columns for X and y
X = data[['RIAGENDR_GENDER', 'BMXBMI_BMI', 'LBXGLU_GLUCOSE', 'DIQ010_DIBETIC', 'LBXGLT_ORAL', 'LBXIN_INSULIN', 'age_group']]
y = data['RIDAGEYR_AGE']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X.iloc[:, :-1]), columns=X.columns[:-1])
X_scaled['age_group'] = X['age_group']  # Preserve the original age_group column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=0)

# Create and train the regression model with more estimators
model = RandomForestRegressor(n_estimators=250, random_state=0)
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred_test = model.predict(X_test)

# Calculate evaluation metrics on the test set
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r_squared_test = r2_score(y_test, y_pred_test)

print("Mean Absolute Error on Test Set:", mae_test)
print("Mean Squared Error on Test Set:", mse_test)
print("R-squared on Test Set:", r_squared_test*100)
print()

# User input for age prediction
while True:
    # Prompt the user for input values
    age_group = le.transform([input("Enter age group (Child, Adult, Senior): ").capitalize()])[0]
    gender = int(input("Enter gender (1 for male, 2 for female): "))
    bmi = float(input("Enter BMI: "))
    glucose = float(input("Enter glucose level: "))
    diabetic = int(input("Enter diabetic status (2 for No, 1 for yes): "))
    oral = float(input("Enter oral: "))
    insulin = float(input("Enter insulin level: "))
    
    # Scale the user input
    user_input_scaled = scaler.transform([[gender, bmi, glucose, diabetic, oral, insulin]])[0]

    # Add age_group back to the scaled input
    user_input_scaled_with_age = [*user_input_scaled, age_group]

    # Make predictions based on user input
    predictions = model.predict([user_input_scaled_with_age])

    # Print the predicted age
    print("Predicted age:", predictions[0])

    another_prediction = input("Do you want to predict another age? (yes/no): ")
    if another_prediction.lower() != "yes":
        break


# In[ ]:




