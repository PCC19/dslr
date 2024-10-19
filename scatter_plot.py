import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

if len(sys.argv) != 2:
    print('Usage: describe.py [file]')
    exit(1)
try:
    df = pd.read_csv(sys.argv[1])
except:
    print('File was not found or it is corrupted')
    exit(1)

print("========= SCATTER PLOT  ================")
print(df.describe())

# =====================================================
# Generate dataframe with numeric columns
# =====================================================
ndf = df.select_dtypes(include='number')
cols = ndf.columns

# =====================================================
# Set up the figure and axes for a grid of subplots
# =====================================================
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16,12))
axes = axes.flatten()

# Scatter Plots
for ax, col in zip (axes[1:] , cols[1:]):
    print(col)
    gfg = sns.scatterplot(x = df.index, y = col, data = df, hue = 'Hogwarts House', ax = ax)
    gfg.set_title(col, fontsize = '8', fontweight = 'bold', pad = 1)
    plt.setp(gfg.get_legend().get_texts(), fontsize='6')
    plt.setp(gfg.get_legend().get_title(), fontsize='6')
    gfg.set_ylabel(ylabel = '', fontsize = 6)
    gfg.set_xlabel(xlabel = '', fontsize = 6)
plt.savefig('scatter.png', bbox_inches='tight')
plt.tight_layout()
plt.show()
