import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def one_hot_encoding(c):
    target_col = c.values # Convert to nympy array
    classes = np.unique(target_col)
    num_classes = len(classes)
    y_binary = np.zeros((target_col.size, num_classes))
    for idx, class_label in enumerate(classes):
        y_binary[:,idx] = (target_col == class_label).astype(int)
    return y_binary

def normalize(c):
    return (c - c.min()) / (c.max() - c.min())

def normalize_df(d):
    out = np.zeros(d.shape)
    for i in range(d.shape[1]):
        out[:,i] = normalize(d[:,i])
    return out

def sigmoid(z):
    return 1 / (1 + np.exp(-z.astype(float)))

def compute_cost(x, y, theta):
    m = y.size
    h = sigmoid(x @ theta)
    cost = (-1 / m) * (y @ np.log(h)) + (1 - y) @ np.log(1 - h)
    return cost

def gradient_descent(x, y, theta, alpha, num_iterations):
    m = y.size
    cost_history = []
    for _ in range(num_iterations):
        h = sigmoid(x @ theta)
        gradient = (1/m) * (x.T @ (h - y)) # Derivative of the cost function
        theta -= alpha * gradient
        cost_history.append(compute_cost(x, y, theta))
    return theta, cost_history

def train_ova(x, y, alpha=0.01, num_iterations=2000, p = False):
    m, n = x.shape
    all_theta = np.zeros((y.shape[1], n))  # No bias term included
    for i in range(y.shape[1]):
        initial_theta = np.zeros(n)
        all_theta[i], cost = gradient_descent(x, y[:, i], initial_theta, alpha, num_iterations)
        print("cost:", cost[-1])
        plt.plot(cost)
        plt.grid()
    plt.draw()
    plt.savefig('cost.png', bbox_inches='tight')
    return all_theta

def predict(x, all_theta):
    print('x:', x.shape, type(x), 't:', all_theta.shape, type(all_theta))
    probabilities = sigmoid(x @ all_theta.T)  
    return np.argmax(probabilities, axis=1) 

# =====================================================
# MAIN
# =====================================================
if len(sys.argv) != 2:
    print('Usage: describe.py [file]')
    exit(1)
try:
    df = pd.read_csv(sys.argv[1])
except:
    print('File was not found or it is corrupted')
    exit(1)

print("========= TRAIN: ================")
df.drop(columns = ['Index'], inplace = True)
df.dropna(inplace = True)
print(df.describe())
print(df)

classes = np.sort(np.unique(df['Hogwarts House']))
print('classes:')
print(classes)

# Generate y
y = one_hot_encoding(df['Hogwarts House'])
print("y:")
print(pd.DataFrame(y))

# Generate x
ndf = df.select_dtypes(include='number')
x = normalize_df(ndf.values)
print('x_norm:')
print(pd.DataFrame(x))
print(x.shape)

# Train model and generate theta file
theta = train_ova(x, y, alpha=0.1, num_iterations=2000, p = True)
print('theta')
table = pd.DataFrame(theta).T
table.index = ndf.columns
table.columns = classes
print(table)
table.to_csv('theta.csv')

# Predict ova
print('x:', x.shape, 't:', theta.shape)
y_pred = predict(x,theta)
print('y_pred:')
print(pd.DataFrame(y_pred))
classes_real = df['Hogwarts House'] 
classes_pred = classes[y_pred]
print('classes_pred:')
print(pd.DataFrame(classes_pred))

# Performance (confusion matrix)
cm = confusion_matrix(classes_real, classes_pred)
cm = cm / classes_real.size * 100
print('Confusion Matrix:')
print(pd.DataFrame(cm))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Logistic Regression Classifier')
plt.draw()
plt.savefig('cm.png', bbox_inches='tight')

# Visualize 4 models
ax = table.plot(kind='bar', title ="Theta",figsize=(12,8),legend=True, fontsize=12)
plt.grid()
plt.tight_layout()
plt.draw()
plt.savefig('models.png', bbox_inches='tight')
plt.show()
