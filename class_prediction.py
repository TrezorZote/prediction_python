import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example dataset
# Features: [Income Level, Credit Score, Age]
data = np.array ([
    [45000, 600, 25],  # Person 1
    [54000, 650, 32],  # Person 2
    [38000, 500, 22],  # Person 3
    [60000, 700, 45],  # Person 4
    [35000, 480, 23],  # Person 5
    [75000, 850, 40],  # Person 6
    [50000, 620, 35],  # Person 7
    [32000, 420, 29],  # Person 8
    [67000, 720, 39],  # Person 9
    [29000, 390, 21],  # Person 10
])

# Labels: 1 means paid on time, 0 means did not pay on time
labels =np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0]) 

# Prepare data (X for features, y for labels)
X = data  # Features: Income level, Credit score, Age
y = labels  # Labels: 1 for paying on time, 0 for not paying on time

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Test the model on the test set
y_pred = model.predict(X_test)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# New person with the following attributes
new_person = [[50000, 650, 30]]

# Predict whether this person will pay on time (1) or not (0)
prediction = model.predict(new_person)

if prediction[0] == 1:
    print("This person is predicted to pay on time.")
else:
    print("This person is predicted to be at risk of not paying on time.")



#Here’s a breakdown of the parameters used in train_test_split(X, y, test_size=0.2, random_state=42):

#X and y:
#X represents the feature set (input data). In the context of your example, X is the list of people's attributes like income level, credit score, and age.
#y represents the labels (output data). These are the corresponding results or classes you are trying to predict (e.g., whether the person paid on time or not).
#test_size=0.2:
#This specifies the proportion of the dataset that should be allocated for testing.
#0.2 means 20% of the data will be used as the test set, and the remaining 80% will be used as the training set.
#You can adjust this value depending on how much data you want to reserve for testing. A typical split is between 70% to 80% for training and 20% to 30% for testing.
#random_state=42:
#The random_state is a seed for the random number generator. Setting this ensures that the split will always be the same every time you run the code, which helps ensure reproducibility.

#train_test_split(X, y, test_size=0.2, random_state=42): gives four outputs x_train x_test y_train y_test
#Input Data (X):
#X contains all the feature data (e.g., attributes like age, income, etc.).
#It is split into X_train and X_test.
#Target Data (y):
#y contains the target values (e.g., labels indicating if the person pays on time or not).
#It is split into y_train and y_test.