import pandas as pd

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
    

# Read csv file in dataframe
df = pd.read_csv('./datasets/dataset_train.csv')
print("Original DataFrame:")
print(df)
print(df.describe())
print(df.info())
df.drop(columns = ['Index'], inplace = True)

# Generate dataframe with numeric columns
numeric_df = df.select_dtypes(include='number')
print("\nNumeric Columns DataFrame:")
print(numeric_df)

res = pd.DataFrame()
dict= {}
for col in numeric_df.columns:
    x = calc_count(numeric_df[col])
    dict.update({'count' : x})
    x = calc_mean(numeric_df[col])
    dict.update({'mean' : x})
    dict.update({'min' : 1})
    temp = pd.DataFrame(dict, index = [col])
    print(dict)
    print(temp)
    res = pd.concat([res,temp])
    print(res)
#    pd.concat([res, pd.DataFrame(dict, index = col#)])
#    res.rename(index={res.index[-1]: col}, inplace=True)

#res = pd.DataFrame(dict.items())
#res = pd.DataFrame([dict])
print(res.transpose())

# Generate dataframe with non-numeric columns
non_numeric_df = df.select_dtypes(exclude='number')
#print("\nNon-Numeric Columns DataFrame:")
#print(non_numeric_df)


#
