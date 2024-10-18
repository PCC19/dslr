import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


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
    return 1 / (1 + np.exp(-z))

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
        gradient = (1/m) * (x.T @ (h - y))
        theta -= alpha * gradient
        cost_history.append(compute_cost(x, y, theta))
    

    return theta, cost_history

def train_ova(x, y, alpha=0.01, num_iterations=2000, p = False):
    m, n = x.shape
    all_theta = np.zeros((y.shape[1], n))  # No bias term included

    for i in range(y.shape[1]):
        initial_theta = np.zeros(n)  # Initialize theta for each class without bias 
        all_theta[i], cost = gradient_descent(x, y[:, i], initial_theta, alpha, num_iterations)
        if p:
            plt.plot(cost)
    if p:
        plt.show()

    return all_theta
# =====================================================
# MAIN
# =====================================================
# Ler df
if len(sys.argv) > 1:
    try:
        # Read csv file in dataframe
        df = pd.read_csv(sys.argv[1])
    except:
        print('File was not found or it is corrupted')
        exit(1)
    
    print("========= TRAIN: ================")
    df.drop(columns = ['Index'], inplace = True)
    df.dropna(inplace = True)
    print(df.describe())
    print(df)

    # Pegar coluna classes
    y = one_hot_encoding(df['Hogwarts House'])
    print("y:")
    print(y)

    # Montar x
    x = normalize_df(df.select_dtypes(include='number').values)
    print('x_norm:')
    print(x)
    print(x.shape)
    theta = train_ova(x, y, alpha=0.01, num_iterations=5000, p = False)
    print('theta')
    print(theta)
    np.savetxt('theta.csv', theta)
    exit(1)
#Para cada classe
#    Montar y
#    Fitar modelo
#    Salvar modelo numa coluna de matriz
#Plotar modelos (thetas)
#Salvar modelo

#uncao previsao
 #   Recebe 4 colunas de pesos
 #   Aplicar regra
 #   Pega melhor
 #   Plota 4 previsoes
    # =====================================================
    # Generate dataframe with numeric columns
    # =====================================================
    ndf = df.select_dtypes(include='number')
    res = pd.DataFrame()
    dict= {}
    for col in ndf.columns:
        x = calc_count(ndf[col])
        dict.update({'count' : x})
       # x = calc_mean(ndf[col])
        dict.update({'mean' : x})
        x = calc_std(ndf[col])
        dict.update({'std' : x})
        x = calc_min(ndf[col])
        dict.update({'min' : x})
        x = calc_percentile(ndf[col], 25)
        dict.update({'25%' : x})
        x = calc_percentile(ndf[col], 50)
        dict.update({'50%' : x})
        x = calc_percentile(ndf[col], 75)
        dict.update({'75%' : x})
        x = calc_max(ndf[col])
        dict.update({'max' : x})
        temp = pd.DataFrame(dict, index = [col])
        res = pd.concat([res,temp])
    print("========= PYTHON DESCRIPTION: ================")
    print(res.transpose())

    # =====================================================
    # Generate dataframe with non-numeric columns
    # =====================================================
    print("============ BONUS: CATEGORICAL STATS ===================")
    cdf = df.select_dtypes(exclude='number')
    print("Categorial Columns:")
    print(cdf.columns.to_list())
    print("="*80)
    dict= {}
    for col in cdf[['Hogwarts House', 'Best Hand']]:
        n, freq = calc_frequency_count(cdf[col])
        freqdf = pd.DataFrame([freq])
        print("Column: ", col)
        print("n distinct: ", n)
        print("Values Frequency:")
        print(freqdf.transpose().to_string(header = False))
        print("="*80)

else:
    print('Usage: describe.py [file]')
