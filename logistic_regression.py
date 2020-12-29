###########Logistic Regression model###########

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

#df1-training data set
#df2-test data set

##############load train data set test dataset
df1 = pd.read_csv('data.csv')            #A0-A16 coulmns-last column for class attribute
df2 = pd.read_csv('testdataNew.csv')    #A0-A15 columns-no column for class attribute

#print(df1.shape)
#print(df2.shape)
#print(df)
#print(df2)

###############handling missing values

#for training data set
df1=df1.replace(['?'],np.NaN)  #replace all missing alues by 'NaN'. otherwise it says there is no missing values
df2=df2.replace(['?'],np.NaN)

#print(df1.shape)
#(df1.isnull().sum())         #get the sum of missinh values in each attribute(column)
#print(df1.isnull().values.sum())  #get the sum of whole missing values

#for test data set
#df2=df2.replace(['?'],np.NaN)

df1=df1.dropna(thresh=15)
df2=df2.dropna(thresh=15)
#print(new_df1.shape)

#print(df1.shape)
#print(df1.isnull().sum())
#print(df1.dtypes['A1'])
#print(df1.dtypes['A9'])
#print(df1.dtypes['A16'])

for col in df1:
    if(df1[col].dtypes == 'object'):#to check the data type
          df1 = df1.fillna(df1[col].value_counts().index[0]) #to get most frequent value and replace it

for col in df2:
    if(df2[col].dtypes == 'object'):#to check the data type
          df2 = df2.fillna(df2[col].value_counts().index[0])

#print(df1.isnull().sum())
#print(df2.isnull().sum())

labelencoder= LabelEncoder()

for col in df1:
    if(df1[col].dtypes == 'object' or df1[col].dtypes == 'bool'):
# Assigning numerical values and storing in another column
       df1[col] = labelencoder.fit_transform(df1[col])

for col in df2:
    if(df2[col].dtypes == 'object' or df2[col].dtypes == 'bool'):
# Assigning numerical values and storing in another column
       df2[col] = labelencoder.fit_transform(df2[col])

#print(df1.head())
#print( )
#print(df2.head())

###############datasets are split into train and test sets

#convert the DataFrame to a NumPy array
df1 = df1.to_numpy()
df2 = df2.to_numpy()

# Segregate features and labels into separate variables(define x,y columns)
x, y = df1[:, 0:15], df1[:, 15]

# Split into train and test sets
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.33, random_state = 0)


################rescaling

#to find MinMaxScaler there should be feature array.not column. so we should convert dataframe into feature array
# Create scaler
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
#all X values are scaled between 0 and 1.
rescaled_xTrain = minmax_scaler.fit_transform(xTrain)
rescaled_xTest = minmax_scaler.fit_transform(xTest)
rescaled_df2 = minmax_scaler.fit_transform(df2)

################logistic libear regression model
#make an instance of the model
# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

#Model is learning the relationship between digits (x_train) and labels (y_train)
logisticRegr.fit(rescaled_xTrain, yTrain)

#predict labels for new data
# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(rescaled_xTest[0].reshape(1,-1))
logisticRegr.predict(rescaled_xTest[0:10])
predictions = logisticRegr.predict(rescaled_xTest)

# Use score method to get accuracy of model
score = logisticRegr.score(rescaled_xTest, yTest)
print(score)




















###############datasets are split into train and test sets





