import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
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
    print('z:',z, z.shape, type(z))
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
        initial_theta = np.zeros(n)  # Initialize theta for each class without bias 
        all_theta[i], cost = gradient_descent(x, y[:, i], initial_theta, alpha, num_iterations)
        print("cost:", cost[-1])
        plt.plot(cost)
        plt.grid()
    plt.draw()
    return all_theta

def predict(x, all_theta):
    print('x:', x.shape, type(x), 't:', all_theta.shape, type(all_theta))
    print(x @ all_theta.T)
    probabilities = sigmoid(x @ all_theta.T)  
    return np.argmax(probabilities, axis=1) 

def load_theta_from_file(file):
    if not os.path.exists(file):
        print("Theta file not found !")
        exit(1)
    thetas = pd.read_csv(file)
    return thetas

def clear_nan(d):
    df = d.copy(deep = True)
    for col in df.columns:
        df[col].fillna(df[col].mean(), inplace = True)
    return df


# =====================================================
# MAIN
# =====================================================
if len(sys.argv) != 3:
    print('Usage: describe.py [file] [thetas]')
    exit(1)
try:
    # Read csv file in dataframe
    df = pd.read_csv(sys.argv[1])
except:
    print('File was not found or it is corrupted')
    exit(1)

print("========= PREDICT: ================")
df.drop(columns = ['Index', 'Hogwarts House'], inplace = True)
#df.dropna(inplace = True)
print(df)
print(df.describe())

# Montar x
ndf = clear_nan(df.select_dtypes(include='number'))
print('ndf')
print(ndf)
print(ndf.describe())
x = normalize_df(ndf.values)
print('x_norm:')
print(pd.DataFrame(x))
print(x.shape, type(x))

# Load theta file
arq = load_theta_from_file(sys.argv[2])
print('arq:', arq)
classes = arq.columns.to_numpy()[1:]
print('clasees:')
print(classes)
theta = arq.values[:,1:].T
print('Theta:')
print(theta, theta.shape, type(theta))

# Predict ova
print('x:', x.shape, 't:', theta.shape)
y_pred = predict(x,theta)
classes_pred = classes[y_pred]
print('y_pred:')
print(pd.DataFrame(y_pred))
print(classes_pred)
