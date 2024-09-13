# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Sample dataset with fruits, shape, color, radius (for demonstration, replace with your data)
data = {
    'Fruit': ['Apple', 'Banana', 'Apple', 'Banana', 'Banana', 'Orange', 'Orange', 'Apple', 'Orange', 'Banana'],
    'Shape': [1, 2, 1, 2, 2, 1, 1, 1, 1, 2],  # 1 = Round, 2 = Elongated
    'Color': [1, 2, 1, 2, 2, 1, 1, 1, 1, 2],  # 1 = Red/Orange, 2 = Yellow
    'Radius': [4, 6, 4, 6, 7, 5, 4, 4, 5, 7]  # Approximate size/radius
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Shape', 'Color', 'Radius']]
y = df['Fruit']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict the results on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


#True Positive (TP): The number of positive instances that were correctly classified as positive by the model.

#Example: Predicting "Banana" correctly when it is actually a banana.
#True Negative (TN): The number of negative instances that were correctly classified as negative by the model.

#Example: Predicting "Not Banana" correctly when it is not a banana.
#False Positive (FP): The number of negative instances that were incorrectly classified as positive (Type I error).

#Example: Predicting "Banana" when it's not actually a banana.
#False Negative (FN): The number of positive instances that were incorrectly classified as negative (Type II error).

#Example: Predicting "Not Banana" when it is actually a banana.

#confusin matrix TP  FN
                #FP TN

#for fruits its like some features are dependent of oneanother that means the naives bayes is not that good
#of a classifier since it works best on independent data


#Consider a classifier that predicts whether a fruit is an apple (positive class) or not (negative class). If we test the classifier with 100 fruits and the confusion matrix is as follows:

                   #Predicted Apple   	        Predicted Not Apple
#Actual Apple	     50	                               10
#Actual Not Apple	  5	                               35

#True Positive (TP) = 50: 50 apples were correctly predicted as apples.
#False Negative (FN) = 10: 10 apples were incorrectly predicted as not apples.
#False Positive (FP) = 5: 5 non-apples were incorrectly predicted as apples.
#True Negative (TN) = 35: 35 non-apples were correctly predicted as not apples.