import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Sample data for 20 fruits
# Attributes: [shape (0=round, 1=elongated), size (cm), color (0=green, 1=yellow, 2=red)]
data = np.array([
    [0, 7, 0],  # Apple (Green)
    [0, 6, 2],  # Apple (Red)
    [0, 8, 2],  # Apple (Red)
    [1, 15, 1],  # Banana (Yellow)
    [1, 14, 1],  # Banana (Yellow)
    [0, 3, 1],  # Lemon (Yellow)
    [0, 4, 1],  # Lemon (Yellow)
    [0, 4, 2],  # Cherry (Red)
    [0, 4, 2],  # Cherry (Red)
    [0, 3, 2],  # Cherry (Red)
    [0, 8, 1],  # Orange (Yellow)
    [0, 9, 1],  # Orange (Yellow)
    [1, 18, 0],  # Cucumber (Green)
    [1, 17, 0],  # Cucumber (Green)
    [0, 10, 2],  # Peach (Red)
    [0, 11, 2],  # Peach (Red)
    [0, 9, 0],  # Grapefruit (Green)
    [0, 8, 0],  # Grapefruit (Green)
    [1, 20, 0],  # Zucchini (Green)
    [1, 19, 0],  # Zucchini (Green)
])

# Labels: 0=Apple, 1=Banana, 2=Lemon, 3=Cherry, 4=Orange, 5=Cucumber, 6=Peach, 7=Grapefruit, 8=Zucchini
labels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Output accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classify a new fruit (e.g., elongated shape, 16 cm size, yellow color -> expected: Banana)
new_fruit = [[1, 16, 1]]
predicted_label = model.predict(new_fruit)
fruit_labels = ["Apple", "Banana", "Lemon", "Cherry", "Orange", "Cucumber", "Peach", "Grapefruit", "Zucchini"]

print(f"Predicted fruit: {fruit_labels[predicted_label[0]]}")
