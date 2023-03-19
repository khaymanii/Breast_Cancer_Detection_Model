
# In[1]:


# Import the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split


# In[2]:


# Data Collection and Processing

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[3]:


print(breast_cancer_dataset)


# In[4]:


# loading data to a dataframe

data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)


# In[5]:


data_frame.head()


# In[6]:


# adding the 'target' column to th dataframe

data_frame['label'] = breast_cancer_dataset.target


# In[7]:


data_frame.tail()


# In[8]:


data_frame.shape


# In[9]:


data_frame.info()


# In[10]:


data_frame.isnull().sum()


# In[11]:


# statistical measures

data_frame.describe()


# In[12]:


# Checking the distribution of target variable

data_frame['label'].value_counts()


# In[13]:


data_frame.groupby('label').mean()


# In[14]:


# Seperating the features and target

X = data_frame.drop(columns='label', axis = 1)
y = data_frame['label']


# In[15]:


print(X)
print(y)


# In[16]:


# Splitting the data into training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[17]:


print(X.shape, X_train.shape, X_test.shape)


# In[18]:


# Standardize the data

from sklearn.preprocessing import StandardScaler


# In[19]:


scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)

X_test_std = scaler.transform(X_test)


# In[20]:


# Building The Neural Network

import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras


# In[21]:


# Setting up the layers of neural networks

model = keras.Sequential([
            keras.layers.Flatten(input_shape=(30,)),
            keras.layers.Dense(20, activation = 'relu'),
            keras.layers.Dense(2, activation='sigmoid')
])


# In[22]:


# Compiling the neural network

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[23]:


# Training the Neural network

history = model.fit(X_train_std, y_train, validation_split=0.1, epochs=10)


# In[26]:


# Visualizing the accuracy and the loss

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')


plt.legend(['training data', 'validation data'], loc = 'lower right')


# In[27]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')


plt.legend(['training data', 'validation data'], loc = 'upper right')


# In[28]:


# Evaluation : Accuracy of the model on test data

loss, accuracy = model.evaluate(X_test_std, y_test)
print(accuracy)


# In[29]:


print(X_test_std.shape)
print(X_test_std[0])


# In[32]:


y_pred = model.predict(X_test_std)


# In[33]:


print(y_pred.shape)
print(y_pred[0])


# In[34]:


# Converting the prediction probability to class labels

y_pred_labels = [np.argmax(i) for i in y_pred]
print(y_pred_labels)


# In[36]:


# Building the predictive system

input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)

# change the input data to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one data point

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data

input_data_std = scaler.transform(input_data_reshape)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 0):
    print('The Tumour is Malignant')
else:
    print('The Tumour is Benign')


# In[ ]:




