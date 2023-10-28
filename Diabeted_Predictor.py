#!/usr/bin/env python
# coding: utf-8

# Pima Imdian Diabetes dataset

# In[7]:


import pandas as pd


# In[11]:


#reading data
pima = pd.read_csv('diabetes.csv')


# In[12]:


pima.head()


# In[13]:


pima.isnull().sum()


# In[17]:


#define X with most frequent class
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age']
X = pima[feature_cols]
y = pima.Outcome


# In[16]:


pima.columns


# In[18]:


#split X and y training and testing sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[26]:


#train a logistic regression model on the training set 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)


# In[27]:


y_pred_class = logreg.predict(X_test)


# In[28]:


from sklearn import metrics


# In[29]:


metrics.accuracy_score(y_test, y_pred_class)


# In[32]:


#examig the class distribution of the testing set
y_test.value_counts()


# In[39]:


type(pd.Series(y_pred_class))


# In[33]:


#precentage of ones
y_test.mean()


# In[34]:


#calculate the percentage of zeroes
1 - y_test.mean()


# In[35]:


#calculate null accuracy (for multiclass classification problem)
y_test.value_counts().head(1)/len(y_test)


# compairing the ture and predicted response values

# In[53]:


#print the first 30 true and predicted response 
print('True:', y_test.values[0:30])
y_pred = pd.Series(y_pred_class)
print('pred:', y_pred.values[0:30])


# Classification accuracy is the easieast classification metric to understand but it doesn't tell about the understyling distribution of response values .

# # Confusion matrix 

# Table that describess the performance of a classification model 

# In[54]:


#IMPORTANT : first argument is true values adn second arg is predicted 
metrics.confusion_matrix(y_test, y_pred_class)


# it's 2*2 matrix because there are 2 response classes 

# Basic Terminolog
# * True Positives(TP): we correctly predicted that they do have diabetes
# * True Negatives(TN): we correctly predicted that they don't have diabetes 
# * False Positives(FP) : we incorrectly predicted that do have diabetes
# * False Negatives(FN) : we incorrectly predicted that don't have diabetes 

# In[60]:


confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]


# ## From confusion matrix
# ##### Classification Accuracy :overall, how often the classifier is correct

# In[62]:


print((TP + TN) /float(TP+TN+FP+FN))
print(metrics.accuracy_score(y_test, y_pred_class))


# #### Classification Error : Overall , how often is the classifier incorrect 

# In[64]:


print((FP+FN)/float(TP+TN+FP+FN))
print(1-metrics.accuracy_score(y_test,y_pred_class))


# #### Sensitivity : when the actual value is positive , how often is the prediction correct 
# * Detection os positive instances
# * Also known as "True Positive Rate" or Recall 

# In[66]:


print(TP/float(TP+FN))
print(metrics.recall_score(y_test, y_pred_class))


# #### Specificity : When the actual value is negative , how often is the prediction correct 
# 

# In[68]:


print(TN/float(TN+FP))


# ### False Positive rate : When actual value is negative , how often is the prediction incorrect

# In[69]:


print(FP/float(TN+FP))


# ### Precision : when a positive value is predicted , how often is the prediction correct

# In[70]:


print(FP/float(FP+TP))


# # Adjusting the classification threshold 

# In[71]:


#print the firstt 10 predicted responses 
logreg.predict(X_test)[0:10]


# In[72]:


#print the first 10 predicted probablities of class member 
logreg.predict_proba(X_test)[0:10, :]


# In[76]:


# store the predicted probablities for class 1
y_pred_prob1 = logreg.predict_proba(X_test)[:, 1]


# In[74]:


#graphical plot
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 


# In[79]:


#histogram of predicted probablities 
plt.hist(y_pred_prob1)
#plt.xlim(0,1)
plt.title('Histogram of predidted probablities')
plt.xlabel('Predicted of probablity of diabetes')
plt.ylabel('Frequency')

