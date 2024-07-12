import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib

# Fetching the MNIST dataset
print("Fetching the MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']
print(f"Dataset shapes - x: {x.shape}, y: {y.shape}")

# Visualizing a sample digit
print("Visualizing a sample digit...")
some_digit = x.iloc[36000]
some_digit_image = some_digit.values.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.title("Sample Digit (Index: 36000)")
plt.axis('off')
plt.show()

# Splitting the data into training and test sets
print("Splitting the data into training and test sets...")
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

# Shuffling the training set
print("Shuffling the training set...")
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

# Creating binary labels for detecting the digit '2'
print("Creating binary labels for detecting the digit '2'...")
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)

# Training a Logistic Regression model
print("Training a Logistic Regression model...")
clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train_2)

# Making a prediction on the sample digit
print("Making a prediction on the sample digit...")
prediction = clf.predict([some_digit])
print(f"Prediction for the sample digit (Index: 36000): {prediction}")

# Evaluating the model using cross-validation
print("Evaluating the model using cross-validation...")
accuracy_scores = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(f"Cross-validation accuracy scores: {accuracy_scores}")
print(f"Mean cross-validation accuracy: {accuracy_scores.mean():.4f}")

# Save results to a file
with open("results.txt", "w") as f:
    f.write(f"Cross-validation accuracy scores: {accuracy_scores}\n")
    f.write(f"Mean cross-validation accuracy: {accuracy_scores.mean():.4f}\n")

print("Results saved to results.txt.")
