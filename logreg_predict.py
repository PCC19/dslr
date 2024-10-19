import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

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

def normalize(c):
    return (c - c.min()) / (c.max() - c.min())

def normalize_df(d):
    out = np.zeros(d.shape)
    for i in range(d.shape[1]):
        out[:,i] = normalize(d[:,i])
    return out

def sigmoid(z):
    return 1 / (1 + np.exp(-z.astype(float)))

def predict(x, all_theta):
    probabilities = sigmoid(x @ all_theta.T)  
    return np.argmax(probabilities, axis=1) 

# =====================================================
# MAIN
# =====================================================
if len(sys.argv) != 3:
    print('Usage: describe.py [file] [thetas]')
    exit(1)
try:
    df = pd.read_csv(sys.argv[1])
except:
    print('File was not found or it is corrupted')
    exit(1)

print("========= PREDICT: ================")
# Read test dataset
df.drop(columns = ['Index', 'Hogwarts House'], inplace = True)
print(df)
print(df.describe())

# Generate x
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
print('theta file:\n', arq)
classes = arq.columns.to_numpy()[1:]
print('classes:\n')
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

# Generate output
output = pd.DataFrame(classes_pred)
output.index.name = 'Index'
output.columns = ['Hogwarts House']
print('output:\n',output)
output.to_csv('houses.csv')
print("="*80)
print("Output saved to houses.csv !!!")
print("="*80)
