#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


df = pd.read_csv("births_by_age.csv")


# In[18]:


df['Lower Age'] = df['AGE'].apply(lambda x: int(x.split('-')[0]))
df['Upper Age'] = df['AGE'].apply(lambda x: int(x.split('-')[1]))
df['Average Age'] = (df['Lower Age'] + df['Upper Age']) / 2


# In[19]:


total_births = df['Count'].sum()

# Calculate success rates
df['Success Rate'] = df['Count'] / total_births
max_success_rate = df['Success Rate'].max()
df['Scaled Success Rate'] = df['Success Rate'] / max_success_rate


# In[20]:


user_age = float(input("Enter age (15-44): "))  # Convert user input to float


# In[27]:


scaled_success_rate = df.loc[(df['Lower Age'] <= user_age) & (df['Upper Age'] >= user_age), 'Scaled Success Rate']

if not scaled_success_rate.empty:
    scaled_success_rate = scaled_success_rate.values[0]
    print(f"Scaled Success Rate for age {user_age}: {scaled_success_rate:.2%}")
else:
    print("Invalid age input.")


# In[28]:


plt.bar(df['Average Age'], df['Scaled Success Rate'])
plt.xlabel('Average Age Group')
plt.ylabel('Scaled Success Rate')
plt.title('Scaled Success Rates of Childbirth by Age Group')
plt.ylim(0, 1.2)  # Adjust the y-axis limit for better visualization
plt.show()


# In[29]:


bmi_categories = {
    (0, 18.5): "Underweight",
    (18.5, 24.9): "Normal Weight",
    (25.0, 29.9): "Overweight",
    (30.0, 40): "Obese"
}


# In[30]:


# Load the dataset
data = pd.read_csv("height_weigt.csv")

# Calculate BMI and categorize into BMI classes
def categorize_bmi(weight, height):
    bmi = (weight / (height / 100) ** 2)
    for bmi_range, label in bmi_categories.items():
        if bmi_range[0] <= bmi < bmi_range[1]:
            return label
    return "Unknown"

data['bmi_category'] = data.apply(lambda row: categorize_bmi(row['weight'], row['height']), axis=1)

# Prepare features and target
X = data[['bmi_category']]
y = data['risk']


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


X_train_encoded = pd.get_dummies(X_train, columns=['bmi_category'])
X_test_encoded = pd.get_dummies(X_test, columns=['bmi_category'])


# In[33]:


model = LogisticRegression()
model.fit(X_train_encoded, y_train)


# In[34]:


# ... (previous code)

# User Input and Prediction
user_weight = float(input("Enter your weight: "))
user_height = float(input("Enter your height: "))

user_bmi_category = categorize_bmi(user_weight, user_height)
print("User BMI Category:", user_bmi_category)

user_data = pd.DataFrame({'bmi_category': [user_bmi_category]})

# Encode the user input data into one-hot encoding
user_data_encoded = pd.get_dummies(user_data, columns=['bmi_category'])

# Add missing columns if necessary
missing_cols = set(X_train_encoded.columns) - set(user_data_encoded.columns)
for col in missing_cols:
    user_data_encoded[col] = 0

# Reorder columns to match training data
user_data_encoded = user_data_encoded[X_train_encoded.columns]

print("User Encoded Data:")
print(user_data_encoded)

predicted_risk = model.predict(user_data_encoded)

if predicted_risk == 0:
    print("Low risk")
else:
    print("High risk")


# In[35]:


sns.scatterplot(x='weight', y='height', hue='risk', data=data, palette={0: 'blue', 1: 'red'})


# In[36]:


# Assume you have obtained scaled_success_rate and high_low_risk from previous calculations

# Define a function to predict the final risk category based on the scaled success rate and high/low risk classification
def predict_final_risk(scaled_success_rate, high_low_risk):
    if high_low_risk == 1:  # High risk
        if scaled_success_rate >= 0.75:
            return "Moderate Risk"
        else:
            return "High Risk"
    else:  # Low risk
        if scaled_success_rate >= 0.5:
            return "Low Risk"
        elif scaled_success_rate <= 0.1:
            return "High Risk"
        else:
            return "Moderate Risk"

# Call the function to predict the final risk
final_risk_prediction = predict_final_risk(scaled_success_rate, predicted_risk)
print("Predicted Final Risk:", final_risk_prediction)


# In[ ]:




