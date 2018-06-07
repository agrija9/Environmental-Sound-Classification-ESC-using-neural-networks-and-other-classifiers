from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

#Load data
X = np.load('feat.npy')
y = np.load('label.npy').ravel()

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize classifier
gnb_clf= GaussianNB() #check input params

# Train model
gnb_clf.fit(X_train, y_train)

# Make predictions
prediction = gnb_clf.predict(X_test)

#print('Predicted values')
#print(prediction)
#print
#print('Actual values')
#print(y_test)

# Evaluate accuracy
print
acc = gnb_clf.score(X_test, y_test)
print("Accuracy = %0.3f" %acc)
#print(accuracy_score(y_test,prediction)) # Equivalent way to do it
