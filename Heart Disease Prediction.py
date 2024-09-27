import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
url = "https://drive.google.com/file/d/1Pnkr_uIwF-uuzWCBgD98_W5jIq0Qe743/view?usp=sharing"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
heart_dataset = pd.read_csv(url)
X = heart_dataset.drop(columns = 'target', axis =1)
Y = heart_dataset['target']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = heart_dataset['target']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Training Accuracy Score: ", training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Testing Accuracy Score: ", test_data_accuracy)
input_data= (67,1,0,120,237,0,1,71,0,1,1,0,2)
input_data_np = np.asarray(input_data)
input_data_reshaped = input_data_np.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
if prediction[0] == 0:
  print("No Heart Disease")
else:
  print("Heart Disease")
