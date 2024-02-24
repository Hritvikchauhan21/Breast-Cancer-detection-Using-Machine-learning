#!/usr/bin/env python
# coding: utf-8

# 
# # Breast Cancer Detection Machine Learning End to End Project Classifier

# # Supervised Learning Classifier Model

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Load

# In[2]:


from sklearn.datasets import load_breast_cancer
cancer_dataset =load_breast_cancer()


# # Data Manupulation

# In[3]:


cancer_dataset


# In[4]:


type(cancer_dataset)


# In[5]:


cancer_dataset.keys()


# In[6]:


cancer_dataset['data']


# In[7]:


type(cancer_dataset['data'])


# In[8]:


# 0 means malignant tumor
# 1 mean benign tumor
cancer_dataset['target']


# In[9]:


cancer_dataset['target_names']


# In[10]:


# The cancer_dataset[‘DESCR’] store the description of breast cancer dataset.
print(cancer_dataset['DESCR'])


# In[11]:


print(cancer_dataset['feature_names'])


# In[12]:


print(cancer_dataset['filename'])


# # Create Dataframe

# In[13]:


cancer_df=pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
                      columns=np.append(cancer_dataset['feature_names'],['target']))


# Pandas DataFrame is two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns. Pandas DataFrame consists of three principal components, the data, rows, and columns.

# In[14]:


cancer_df.to_csv('breast_cancer_dataframe.csv')


# In[15]:


cancer_df.head(6)


# In[16]:


cancer_df.tail(6)


# In[17]:


cancer_df.info()


# In[18]:


cancer_df.describe()


# In[19]:


cancer_df.isnull().sum()


# # Data Visualization

# In[20]:


sns.pairplot(cancer_df,hue='target')


# In[21]:


#pair plotof smaple feature
sns.pairplot(cancer_df,hue='target',
            vars=['mean radius','mean texture','mean perimeter','mean area','mean smoothness'])


# In[22]:


sns.countplot(x='target',data=cancer_df)


# In[23]:


plt.figure(figsize=(20,8))
sns.countplot(x='mean radius',data=cancer_df) #*********ing 7*********


# # Heatmap

# Heatmap of breast cancer DataFrame
# In the below heatmap we can see the variety of different feature’s value. The value of feature ‘mean area’ and ‘worst area’ are greater than other and ‘mean perimeter’, ‘area error’, and ‘worst perimeter’ value slightly less but greater than remaining featur

# In[24]:


# heatmap of dataframe
plt.figure(figsize=(16,9))
sns.heatmap(cancer_df)


# # Heatmap of a correlation matrix

# To find a correlation between each feature and target we visualize heatmap using the correlation matrix.

# In[25]:


# Heatmap of correlation matrix of breast cancer Dataframe 
plt.figure(figsize=(20,20))
sns.heatmap(cancer_df.corr(),annot=True,cmap='coolwarm',linewidths=2)


# # Correlation barplot

# Taking the correlation of each feature with the target and the visualize barplot.

# In[26]:


# create second Dataframe by droping target
cancer_df2=cancer_df.drop(['target'],axis=1)
print("The shape of cancer_df2 is:",cancer_df2.shape)
print("The shape of cancer_dfis:",cancer_df.shape)


# In[27]:


# visualize correlation barplot
plt.figure(figsize=(16,5))
ax = sns.barplot(x='cancer_df2.corrwith(cancer_df.target).index', y='cancer_df2.corrwith(cancer_df.target)',data=cancer_df2)
# ax=sns.barplot(x='cancer_df2.corr(cancer_df.target)' , y='cancer_df2.corr(cancer_df.target).index',data=cancer_df2)
ax.tick_params(labelrotation=90)


# # Data Processing 

# Split DataFrame in train And test

# In[28]:


# input variable
X=cancer_df.drop(['target'],axis=1)
X.head(6)


# In[29]:


# output variable
y=cancer_df['target']
y.head(6)


# In[30]:


# split dadtset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test= train_test_split(X,y, test_size=0.2,random_state=5)


# # Feature Scaling

# # Converting different units and magnitude data in one unit.

# In[31]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.transform(X_test)


# Breast Cancer Detection Machine Learning Model Building
# We have clean data to build the Ml model. But which Machine learning algorithm is best for the data we have to find. The output is a categorical format so we will use supervised classification machine learning algorithms.
# 
# To build the best model, we have to train and test the dataset with multiple Machine Learning algorithms then we can find the best ML model. So let’s try.
# 
# First, we need to import the required packages.

# In[32]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# # Support Vector Classifier

# In[33]:


# supoort vector classifier
from sklearn.svm import SVC
svc_classifier=SVC()
svc_classifier.fit(X_train,y_train)
y_pred_scv=svc_classifier.predict(X_test)
accuracy_score(y_test,y_pred_scv)


# In[34]:


# train with standard scaled data
svc_classifier2=SVC()
svc_classifier2.fit(X_train_sc,y_train)
y_pred_svc_sc=svc_classifier2.predict(X_test_sc)
accuracy_score(y_test,y_pred_svc_sc)


# # Logistic Regression

# In[35]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier=LogisticRegression(random_state=51,penalty='l2')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_score(y_test, y_pred_lr)


# In[36]:


# Train with Standard scaled Data
lr_classifier2 = LogisticRegression(random_state = 51, penalty = 'l2')
lr_classifier2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_lr_sc)


# # K – Nearest Neighbor Classifier

# In[37]:


# k-Nearest neighbor classifier
# from sklearn.neighbors import KNeighborsClassifier
# knn_classifier=KNeighborsClassifier(n_neighbors = 5,metric='minkowski',p=2)
# knn_classifier.fit(X_train, y_train)
# y_pred_knn=knn_classifier.predict(X_test)
# accuracy_score(y_test, y_pred_knn)
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)


# In[38]:


# Train with standard scaled data
knn_classifier2=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn_classifier2.fit(X_train_sc,y_train)
y_pred_knn_sc=knn_classifier.predict(X_test_sc)
accuracy_score(y_test,y_pred_knn_sc)


# # XGBoost Parameter Tuning Randomized Search

# In[39]:


from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_xgb)


# In[ ]:


pip install xgboost


# In[40]:


xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)


# # Confusion Matrix

# In[41]:


cm=confusion_matrix(y_test,y_pred_xgb)
plt.title('Heatmap of confusion Matrix',fontsize=15)
sns.heatmap(cm,annot=True)
plt.show()


# In[42]:


print(classification_report(y_test,y_pred_xgb))


# # Cross valdiation of the ML model 

# To find the Ml is overfitted ,under fitted or d=generalize doing cross-valdiation

# In[44]:


from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_classifier, X = X_train_sc, y = y_train, cv = 10)
print("Cross validation of XGBoost model = ",cross_validation)
print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_classifier, X = X_train_sc,y = y_train, cv = 10)
print("Cross validation accuracy of XGBoost model = ", cross_validation)
print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())


# # Save the Machine learning Model

# In[45]:


# import pickle
# # save the model
# pickle.dump(xgb_classifier_pt, open('breast_cancer_detector.pickle','wb'))
# # load model
# breast_cancer_detector_model=pickle.load(open('breast_cancer_detector.pickle','rb'))
# # predict the output
# y_pred= breast_cancer_detector_model.predict(X_test)
# # confusion matrix
# print('confusion matrix of XGBoost model:\n',confusion_matrix(y_test,y_pred),'\n')
# # show the accuracy
# print('Accuracy of XGBoost model =',accuracy_score(y_test,y_pred))
## Pickle
import pickle
 
# save model
pickle.dump(xgb_classifier, open('breast_cancer_detector.pickle', 'wb'))
 
# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
 
# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)
 
# confusion matrix
print('Confusion matrix of XGBoost model: \n',confusion_matrix(y_test, y_pred),'\n')
 
# show the accuracy
print('Accuracy of XGBoost model = ',accuracy_score(y_test, y_pred))

