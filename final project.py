#!/usr/bin/env python
# coding: utf-8

# In[223]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


# In[224]:


# Extract
nearest_objects = pd.read_csv('nearest-earth-objects(1910-2024).csv')
nearest_objects


# In[225]:


nearest_objects.head(10)


# In[226]:


nearest_objects.tail(10)


# In[227]:


nearest_objects.sample()


# In[228]:


nearest_objects.shape


# In[229]:


#  Q2
#clean
nearest_objects.isnull().sum() 


# In[230]:


nearest_objects.dropna(inplace=True) #remove null


# In[231]:


nearest_objects.isnull().sum() 


# In[232]:


nearest_objects.duplicated().sum()


# In[233]:


nearest_objects.shape


# In[234]:


#Inconsistency
nearest_objects = nearest_objects[nearest_objects['estimated_diameter_min'] >= 0]
nearest_objects = nearest_objects[nearest_objects['estimated_diameter_max'] >= 0]
nearest_objects = nearest_objects[nearest_objects['miss_distance'] >= 0]


# In[235]:


nearest_objects.shape


# In[236]:


#Inconsistency (اول حرف من كل كلمه كبير)
nearest_objects['is_hazardous'] = [x.title() if isinstance(x, str) else x for x in nearest_objects['is_hazardous']]
nearest_objects


# In[237]:


#Inconsistency
nearest_objects['orbiting_body'] = [x.title() if isinstance(x, str) else x for x in nearest_objects['orbiting_body']]
nearest_objects


# In[238]:


nearest_objects.info()


# In[239]:


nearest_objects.describe(include="all") # describe all data


# In[240]:


nearest_objects.describe(include="number") # describe all numeric data


# In[241]:


nearest_objects.describe(include="object")  # describe all non numeric data


# In[242]:


# relationship between col (visualization) ( before remove outliers)
plt.subplot(1, 2, 1)
counts = nearest_objects['is_hazardous'].value_counts()
colors = ['pink','skyblue']  # Add more colors as needed
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors,startangle=140,shadow=True)
plt.title('Is Hazardous Frequency')
plt.legend(title='Is Hazardous',loc='lower right')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.subplot(1, 2, 2)
counts = nearest_objects['orbiting_body'].value_counts()
colors = ['lightgreen']  # Add more colors as needed
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors,startangle=140,shadow=True)
plt.title('Orbiting Body Frequency')
plt.legend(title='Orbiting Body',loc='lower right')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


# In[243]:


#العلاقه بين القطرين
plt.figure(figsize=(8, 6))
plt.scatter(nearest_objects['estimated_diameter_min'], nearest_objects['estimated_diameter_max'], alpha=0.3,color='pink')
plt.xlabel('estimated_diameter_min')
plt.ylabel('estimated_diameter_max')
plt.title('Diameter distribution')
plt.show()


# In[244]:


# العلاقه بين الحجم والقطر لكل قطر
# Set the figure size
plt.figure(figsize=(8, 5))
# Create the scatter plot
sns.scatterplot(data=nearest_objects, x='estimated_diameter_min', y='absolute_magnitude', alpha=0.3)
# Set labels and title
plt.xlabel('estimated_diameter_min')
plt.ylabel('magnitude')
plt.title('Relationship between diameter_min and magnitude')
plt.show()

# Set the figure size
plt.figure(figsize=(8, 5))
# Create the scatter plot
sns.scatterplot(data=nearest_objects, x='estimated_diameter_max', y='absolute_magnitude', alpha=0.3)
# Set labels and title
plt.xlabel('estimated_diameter_max')
plt.ylabel('magnitude')
plt.title('Relationship between diameter_max and magnitude')
plt.show()


# In[245]:


# frequency of some cols.
plt.figure(figsize=(7, 5))
plt.hist(nearest_objects['absolute_magnitude'], bins=30, color='seagreen')
plt.title('Distribution of Absolute Magnitude')
plt.xlabel('Absolute Magnitude')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(nearest_objects['estimated_diameter_max'], bins=30, color='lightgreen')
plt.title('Distribution of Estimated Max Diameter')
plt.xlabel('Max Diameter (km)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(nearest_objects['estimated_diameter_min'], bins=30, color='lightblue')
plt.title('Distribution of Estimated Min Diameter')
plt.xlabel('Min Diameter (km)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(nearest_objects['miss_distance'], bins=30, color='pink')
plt.title('Distribution of miss_distance')
plt.xlabel('miss_distance (km)')
plt.ylabel('Frequency')
plt.show()


# In[246]:


#correlation 
features = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']
corr_matrix = nearest_objects[features].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(np.arange(len(features)), features, rotation=45)
plt.yticks(np.arange(len(features)), features)
plt.title('Correlation Matrix')
plt.show()


# In[247]:


# العلاقة بين السرعة والقطرين
plt.figure(figsize=(7, 5))
plt.scatter(nearest_objects['relative_velocity'], nearest_objects['estimated_diameter_max'], color='yellow', alpha=0.5)
plt.title('Relative Velocity vs Estimated Max Diameter')
plt.xlabel('Relative Velocity (km/h)')
plt.ylabel('Max Diameter (km)')
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(nearest_objects['relative_velocity'], nearest_objects['estimated_diameter_min'], color='purple', alpha=0.5)
plt.title('Relative Velocity vs Estimated Min Diameter')
plt.xlabel('Relative Velocity (km/h)')
plt.ylabel('Min Diameter (km)')
plt.show()


# In[248]:


#العلاقه بين السرعه والمسافه 
plt.figure(figsize=(7, 5))
plt.scatter(nearest_objects['miss_distance'], nearest_objects['relative_velocity'], color='red', alpha=0.5)
plt.title('Miss Distance vs Relative Velocity')
plt.xlabel('Miss Distance (km)')
plt.ylabel('Relative Velocity (km/h)')
plt.show()


# In[249]:


#العلاقه بين الحجم والمسافه 
plt.figure(figsize=(7, 5))
plt.scatter(nearest_objects['miss_distance'], nearest_objects['absolute_magnitude'], color='lightgreen', alpha=0.5)
plt.title('Miss Distance vs Magnitude')
plt.xlabel('Miss Distance (km)')
plt.ylabel('Absolute Magnitude (km/h)')
plt.show()


# In[250]:


fig2 = go.Figure(data=[go.Scatter3d(
    x=nearest_objects['estimated_diameter_max'],
    y=nearest_objects['estimated_diameter_min'],
    z=nearest_objects['relative_velocity'],
    mode='markers',
    marker=dict(
        size=5,
        color=nearest_objects['relative_velocity'],
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig2.update_layout(scene=dict(
                    xaxis_title='diameter_max',
                    yaxis_title='diameter_min',
                    zaxis_title='relative_velocity'),
                    title='diameter_max vs diameter_min vs relative_velocity')
fig2.show()


# In[279]:


# outliers
def remove_outliers(nearest_objects, feature):
    Q1 = nearest_objects[feature].quantile(0.25)
    Q3 = nearest_objects[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return nearest_objects[(nearest_objects[feature] >= lower) & (nearest_objects[feature] <= upper)]

print(f'Original shape: {nearest_objects.shape}')


for column in ['absolute_magnitude', 'estimated_diameter_min', 
               'estimated_diameter_max', 'relative_velocity', 'miss_distance']:
    nearest_objects = remove_outliers(nearest_objects, column)

print(f'Cleaned shape: {nearest_objects.shape}')


# In[287]:


# visualization (after remove outliers)
#العلاقه بين القطرين
plt.figure(figsize=(8, 6))
plt.scatter(nearest_objects['estimated_diameter_min'], nearest_objects['estimated_diameter_max'], alpha=0.3,color='pink')
plt.xlabel('estimated_diameter_min')
plt.ylabel('estimated_diameter_max')
plt.title('Diameter distribution')
plt.show()


# In[288]:


# العلاقه بين الحجم والقطر لكل قطر
# Set the figure size
plt.figure(figsize=(8, 5))
# Create the scatter plot
sns.scatterplot(data=nearest_objects, x='estimated_diameter_min', y='absolute_magnitude', alpha=0.3)
# Set labels and title
plt.xlabel('estimated_diameter_min')
plt.ylabel('magnitude')
plt.title('Relationship between diameter_min and magnitude')
plt.show()

# Set the figure size
plt.figure(figsize=(8, 5))
# Create the scatter plot
sns.scatterplot(data=nearest_objects, x='estimated_diameter_max', y='absolute_magnitude', alpha=0.3)
# Set labels and title
plt.xlabel('estimated_diameter_max')
plt.ylabel('magnitude')
plt.title('Relationship between diameter_max and magnitude')
plt.show()


# In[289]:


# frequency some cols.
plt.figure(figsize=(7, 5))
plt.hist(nearest_objects['absolute_magnitude'], bins=30, color='seagreen')
plt.title('Distribution of Absolute Magnitude')
plt.xlabel('Absolute Magnitude')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(nearest_objects['estimated_diameter_max'], bins=30, color='lightgreen')
plt.title('Distribution of Estimated Max Diameter')
plt.xlabel('Max Diameter (km)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(nearest_objects['estimated_diameter_min'], bins=30, color='lightblue')
plt.title('Distribution of Estimated Min Diameter')
plt.xlabel('Min Diameter (km)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(nearest_objects['miss_distance'], bins=30, color='pink')
plt.title('Distribution of miss_distance')
plt.xlabel('miss_distance (km)')
plt.ylabel('Frequency')
plt.show()


# In[268]:


# Feature selection
features = nearest_objects.drop(['neo_id', 'name', 'is_hazardous'], axis=1)
target = nearest_objects['is_hazardous']


# In[269]:


# Encoding categorical 
features = pd.get_dummies(features, drop_first=True)
# Scaling numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[270]:


# Splitting
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)


# In[271]:


# Model training with class weight adjustment
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)


# In[272]:


#   Q4
# Predictions and evaluation
y_pred = model.predict(X_test)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f'Balanced Accuracy: {balanced_accuracy:.2f}')


# In[284]:


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42)
}

# Train models and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Check if the value is a dictionary
            precision = metrics.get('precision')  # avoid KeyError
            auc = roc_auc_score(y_test, y_proba)

            results[name] = {
                f'Precision ({label})': precision,
                'AUC-ROC': auc
            }


# In[285]:


# Display results
results_df = pd.DataFrame(results).T
print(results_df)


# In[286]:


# Plot AUC-ROC curves
plt.figure(figsize=(10, 6))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve')
plt.legend()
plt.show()

