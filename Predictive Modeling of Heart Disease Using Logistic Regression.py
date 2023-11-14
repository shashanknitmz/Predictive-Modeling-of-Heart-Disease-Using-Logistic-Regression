#!/usr/bin/env python
# coding: utf-8

# # Predictive Modeling of Heart Disease Using Logistic       Regression and Data Visualization Techniques

# ## Importing the libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing and Reading the dataset

# In[3]:


df = pd.read_csv("C:/Users/shash/Downloads/framingham.csv")
df.head(10)


# ## Analysis of Data

# In[4]:


df.shape


# In[5]:


df.keys()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# ## Removing NaN / NULL vales from the data

# In[9]:


df.dropna(axis = 0, inplace = True) 
print(df.shape)


# In[10]:


df['TenYearCHD'].value_counts()


# ## Data Visualization

# ### Correlation Matrix

# In[11]:


plt.figure(figsize = (14, 10)) 
sns.heatmap(df.corr(), cmap='Purples',annot=True, linecolor='Green', linewidths=1.0)
plt.show()


# ## Pairplot

# In[12]:


sns.pairplot(df)
plt.show()


# ## Countplot of people based on their sex and whether they are Current Smoker or not

# In[13]:


sns.catplot(data=df, kind='count', x='male',hue='currentSmoker')
plt.show()


# ## Countplot - subplots of No. of people affecting with CHD on basis of their sex and current smoking.

# In[14]:


sns.catplot(data=df, kind='count', x='TenYearCHD', col='male',row='currentSmoker', palette='Blues')
plt.show()


# # Machine Learning Part

# ## Separating the data into feature and target data.

# In[15]:


X = df.iloc[:,[1,10,14]]
y = df.iloc[:,15:16]


# In[16]:


X.head()


# In[17]:


print(y)


# ## Importing the model and assigning the data for training and test set

# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21)


# ## Applying the ML model - Logistic Regression

# In[19]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=5000)


# ## Training the data

# In[20]:


logreg.fit(X_train, y_train.values.ravel())


# ## Testing the data

# In[21]:


y_pred = logreg.predict(X_test)


# ## Predicting the score

# In[22]:


score = logreg.score(X_test, y_test)
print("Prediction score is:",score) 


# ## Getting the Confusion Matrix and Classification Report

# ## Confusion Matrix

# In[23]:


from sklearn.metrics import confusion_matrix, classification_report 
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix is:\n",cm)


# ## Classification Report

# In[24]:


print("Classification Report is:\n\n",classification_report(y_test,y_pred))


# ## Plotting the confusion matrix

# In[25]:


conf_matrix = pd.DataFrame(data = cm,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (10, 6)) 
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens", linecolor="Black", linewidths=1.5) 
plt.show() 


# In[ ]:




