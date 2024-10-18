import pandas as pd
import math
import sys

# All functions receive a dataframe column and return a number
def calc_count(c):
    n = 0
    for row in c:
        if not pd.isna(row):
            n += 1
    return n

def calc_mean(c):
    n = calc_count(c)
    total = 0
    for row in c:
        if not pd.isna(row):
            total += row
    return total / n
    
def calc_std(c):
    n = calc_count(c)
    m = calc_mean(c)
    total = 0;
    for row in c:
        if not pd.isna(row):
            total += (row - m)**2
    return math.sqrt(total/(n-1))

def calc_min(c):
    min = float('inf')
    for row in c:
        if not pd.isna(row):
            if row < min:
                min = row
    return min

def calc_max(c):
    max = float('-inf')
    for row in c:
        if not pd.isna(row):
            if row > max:
                max = row
    return max

def calc_percentile(c, p):
    cc = sorted(c.dropna())
    n = calc_count(cc)
    rank = p / 100 * (n - 1)
    ant = int(rank)
    frac = rank - ant
    if (frac == 0):
        pos = ant
    else:
        pos = ant + 1
    percentile = cc[ant] + frac * (cc[pos] - cc[ant])
    #print ("n: ", n, " rank:", rank, "ant:", ant, " pos: ", pos, " frac:", frac, "cc_ant: ", cc[ant], "cc_pos", cc[pos], " p:", percentile)
    return percentile
    
# Bonus function for categorical data. Returns a number and a dictionary
def calc_frequency_count(c):
    dic= {}
    for row in c:
        if row in dic:
            dic[row] += 1
        else:
            dic[row] = 1
    return len(dic), dic


# =====================================================
# MAIN
# =====================================================
if len(sys.argv) != 2:
    print('Usage: describe.py [file]')
    exit(1)

try:
    # Read csv file in dataframe
    df = pd.read_csv(sys.argv[1])
except:
    print('File was not found or it is corrupted')
    exit(1)

#df.drop(columns = ['Index'], inplace = True)
print("========= MY DESCRIPTION: ================")
print(df.describe())

# =====================================================
# Generate dataframe with numeric columns
# =====================================================
ndf = df.select_dtypes(include='number')
res = pd.DataFrame()
dict= {}
for col in ndf.columns:
    x = calc_count(ndf[col])
    dict.update({'count' : x})
    x = calc_mean(ndf[col])
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
