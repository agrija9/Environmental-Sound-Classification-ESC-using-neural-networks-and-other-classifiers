# coding= UTF-8
from sklearn.ensemble import RandomForestClassifier #Random Forest classifier
import pandas as pd
import numpy as np
np.random.seed(0)

#Load data
X = np.load('feat.npy')
y = np.load('label.npy').ravel()

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Initialize classifier
rf_clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train model
rf_clf.fit(X_train, y_train)

# Make predictions
y_prediction = rf_clf.predict(X_test)

#print('Predicted values')
#print(y_prediction)
#print
#print('Actual values')
#print(y_test)
#print
#print(y_prediction-y_test)

# Evaluate accuracy
print
acc = rf_clf.score(X_test, y_test)
print("Accuracy = %0.3f" %acc)

# View the predicted probabilities of the first n observations
rf_clf.predict_proba(X_test)[0:10]

# For  label decoding
label_classes = np.array(['Dog bark','Rain','Sea waves','Baby cry','Clock tick','Person sneeze','Helicopter','Chainsaw','Rooster',
                          'Fire crackling'])
#print(label_classes)

# Decoding predicted and actual classes (numeric to written)
prediction_decoded = label_classes[y_prediction]
actual_value_decoded = label_classes[y_test]

## Generate Confusion Matrix
pd.crosstab(actual_value_decoded, prediction_decoded)

