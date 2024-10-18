import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

if len(sys.argv) != 2:
    print('Usage: describe.py [file]')
    exit(1)
try:
    # Read csv file in dataframe
    df = pd.read_csv(sys.argv[1])
except:
    print('File was not found or it is corrupted')
    exit(1)

df.drop(columns = ['Index'], inplace = True)
print("========= PAIR PLOT  ================")
print(df.describe())

# =====================================================
# Generate pairplot
# =====================================================
gfg = sns.pairplot(df, hue = 'Hogwarts House', diag_kind="kde")
plt.savefig('pair_plot.png', bbox_inches='tight')
plt.tight_layout()
plt.show()
