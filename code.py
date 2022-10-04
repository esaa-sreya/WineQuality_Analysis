#Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#Step 2: Load Data
dataset = pd.read_csv('D:\esa\PROJECTS\WineQuality_Analysis\Red_wine_dataset.csv')
dataset.shape
dataset.head


#Step 3: Checking Missing Values
dataset.isnull().sum()

dataset = dataset.dropna(axis=0)


#Step 4: Data Analysis and Visualization
dataset.describe()

#Number of values for each quality
sns.catplot(x= "quality", data = dataset, kind= "count")

#volatile acidity vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x= "quality", y= "volatile acidity", data = dataset)

#citric acid vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x= "quality", y= "citric acid", data = dataset)

#Correlation
correlation = dataset.corr()

#Constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar= True, square= True, fmt= '.1f', annot= True, annot_kws= {"size": 8}, cmap = "Blues")


#Step 5: Data Preprocessing

#Separate the data and label
X = dataset.drop('quality', axis = 1)
print(X)

#Label Binarization
Y = dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(Y)


#Step 6: Train and Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)
print(Y.shape, Y_train.shape, Y_test.shape)


#Step 7: Training Model: Rnadom Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)


#Step 8: Model Evaluation

#Accuracy Score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy: ', test_data_accuracy)

predicted_df = {'predicted_values': X_test_prediction, 'original_values': Y_test}

#Creating new dataframe
print(pd.DataFrame(predicted_df).head(20))

#Biulding a predictive system
input_data = (11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8)

#Changing input data to numpy array
input_data_np = np.array(input_data)

#Reshape the data as we are predicting the label for only one value
input_data_reshaped = input_data_np.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if( prediction[0] == 1):
  print('Result of Prediction for specific input data: Good Quality Wine')
else:
  print('Result of Prediction for specific input data: Bad Quality Wine')

