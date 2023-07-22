#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore Data Science Internship Program (July-2023)

# # Presented By:  Aviral Jain

# # Beginner Level Task

# # TASK 1-- Iris Flower Classification Machine learning Project

# # 1- Import Necessary Library

# In[58]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # 2- Reading The DataSet

# In[59]:


my_data = pd.read_csv("D:/DOWNLOAD/Iris.csv")


# # 3-Viewing The Dataset

# In[70]:


my_data.head(10)


# In[71]:


my_data.info()


# In[75]:


my_data.describe()


# In[76]:


my_data.tail(10)


# In[77]:


dir(my_data)


# In[78]:


my_data.shape


# # 4-Data Selection and filtering:

# In[79]:


my_data['SepalLengthCm']
# to access a specific column in the dataframe.


# In[80]:


my_data.loc[1]


# In[81]:


my_data.iloc[2]


# In[82]:


filtered_rows = my_data.loc[my_data['PetalLengthCm'] > 1.9]
print(filtered_rows)


# # 5-Data Manipulation:

# In[83]:


my_data.drop(columns = ['Id'])


# In[84]:


my_data.drop(columns = ['SepalLengthCm','SepalWidthCm'])


# In[85]:


my_data.rename(columns={'Species':'Variety'})


# In[86]:


# Grouping the data based on the 'Species' column
grouped_data = my_data.groupby('Species')

# You can now apply aggregate functions on the grouped data, such as mean, sum, etc.
# For example, let's calculate the mean values for each group:
mean_values = grouped_data.mean()

print(mean_values)


# In[87]:


grouped_data = my_data.groupby('Species')
sum_values = grouped_data.sum()
print(sum_values)


# In[88]:


# Sorting the DataFrame based on the 'PetalLengthCm' column in ascending order
sorted_data = my_data.sort_values(by='PetalLengthCm', ascending=True)

print(sorted_data)


# 
# 
# # 6-Data analysis
# 

# In[89]:


# To count the occurrences of unique values in a column
my_data['Species'].value_counts()


# In[90]:


my_data['PetalLengthCm'].value_counts()


# In[91]:


my_data['PetalLengthCm'].mean()


# In[92]:


my_data['SepalWidthCm'].median()


# In[93]:


my_data.max()


# In[94]:


my_data.min()


# In[95]:


my_data.std()


# In[96]:


# Assuming df is your DataFrame containing the Iris dataset
correlation_matrix = my_data.corr()

print(correlation_matrix)


# In[97]:


# Assuming 'PetalLengthCm' is the column of interest
correlation_with_petal_length = my_data.corr()['PetalLengthCm']

print(correlation_with_petal_length)


# In[98]:


my_data['SepalLengthCm'].plot(kind='box')


# In[99]:


my_data.plot.scatter(x='PetalLengthCm',y='SepalLengthCm')


# In[100]:


my_data.isnull()


# In[101]:


my_data.isnull().sum()


# In[102]:


my_data.duplicated()


# In[103]:


my_data.duplicated().sum()


# # 7-Exploratory Data Analysis

# In[104]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your dataset loaded into 'my_data' DataFrame

# Create a pair plot with hue="Species"
sns.pairplot(my_data, hue="Species")

# Show the plot
plt.show()


# In[105]:


sns.pairplot(my_data, hue = "Species" , kind ='reg')
     


# In[106]:


import matplotlib.pyplot as plt


# # Scatter plot for Sepal Length vs. Sepal Width

# In[112]:




plt.scatter(my_data['SepalLengthCm'], my_data['SepalWidthCm'],  cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset: Sepal Length vs. Sepal Width')
plt.colorbar(label='Species')
plt.show()


# # 8-Machine Learning Model

# In[108]:




# Machine learning model selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# For feature engineering and selection
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score    #to measure model performance
from sklearn.preprocessing import LabelEncoder

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[109]:


plt.figure(figsize=(16, 4))
variables = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalLengthCm']
colors = ['blue', 'purple', 'green', 'indigo']

for i, variable in enumerate(variables):
    plt.subplot(1, 4, i+1)
    sns.boxplot(data=my_data, y=variable, color=colors[i])
    plt.title(variable)

plt.tight_layout()
plt.show()


# In[116]:



plt.figure(figsize=(32, 4))
import matplotlib.pyplot as plt
import seaborn as sns

variables = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

for variable in variables:
    plt.figure(figsize=(6, 4))
    sns.violinplot(y='Species', x=variable, data=my_data, inner='quartile')
    plt.show()


# In[111]:


plt.figure(figsize=(5,5))
sns.heatmap(my_data.corr(), annot=True,cmap='crest')
plt.show()


# # 9-Outlier Detection 

# In[42]:


##using IQR to define the code for outlier detection and percentage.

Q1 = my_data['SepalWidthCm'].quantile(0.25)
Q3 = my_data['SepalWidthCm'].quantile(0.75)
IQR = Q3 - Q1

print("Quartile 1:", Q1)
print("Quartile 3:", Q3)
print("Interquartile Range:", IQR)

upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR

print("Upper Bound:", upper)
print("Lower Bound:", lower)

outliers_upper = my_data[my_data['SepalWidthCm'] > upper]
outliers_lower = my_data[my_data['SepalWidthCm'] < lower]

print("Number of Outliers (Upper):", outliers_upper.shape[0])
print("Number of Outliers (Lower):", outliers_lower.shape[0])


# In[43]:


print("Before Removing Outliers : ", my_data.shape)
data = my_data.drop(index= outliers_upper.shape[0])
data = my_data.drop(index= outliers_lower.shape[0])


# In[44]:


print("Before Removing Outliers:", my_data.shape)

data = my_data[(my_data['SepalWidthCm'] >= lower) & (my_data['SepalWidthCm'] <= upper)]

print("After Removing Outliers:", data.shape)


# In[45]:


data.plot(kind = 'box',subplots = True, layout =(2,3) ,title = 'After Removing Outliers')


# #  10-Label Ecoder

# in machine learning we usually deal with datasets which contains multiple labels in one or more than 
# one columns. These labels can be in the form of words or numbers. Label Ecoding refers to converting the 
# labels into numeric form so as to convert it into the machine readable form
# 

# In[46]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# In[47]:


my_data["Species"] = le.fit_transform(my_data["Species"])
my_data.head(6)
#my_data.tail(6)


# # 11-Model Traning

# In[48]:


from sklearn.model_selection import train_test_split
#train_70% data
#test_30% data
X = my_data.drop(columns=["Species"])
Y = my_data["Species"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.30)


# # 12-Logistic Regression

# In[49]:


#Logistic Regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[50]:


model.fit(x_train,y_train)


# In[51]:


#Print metric to get perfomance 
print("Accuracy:",model.score(x_test,y_test)*100)


# # 13-KNN-k-nearest neighbours
# 

# In[52]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[53]:


model.fit(x_train,y_train)


# In[54]:


#Print metric to get perfomance 
print("Accuracy:",model.score(x_test,y_test)*100)


# # 14-Decision Tree

# In[55]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[56]:


model.fit(x_train,y_train)


# In[57]:


#Print metric to get perfomance 
print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




