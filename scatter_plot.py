
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

if len(sys.argv) > 1:
    try:
        # Read csv file in dataframe
        df = pd.read_csv(sys.argv[1])
    except:
        print('File was not found or it is corrupted')
        exit(1)
    
    #df.drop(columns = ['Index'], inplace = True)
    print("========= HISTOGRAM  ================")
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
    axes = axes.flatten() # Flatten to easily iterate over

    # Scatter Plots
    for ax, col in zip (axes[1:] , cols[1:]):
        print(col)
        gfg = sns.scatterplot(x = df.index, y = col, data = df, hue = 'Hogwarts House', ax = ax)
        gfg.set_title(col, fontsize = '8', fontweight = 'bold', pad = 1)
        plt.setp(gfg.get_legend().get_texts(), fontsize='6')
        plt.setp(gfg.get_legend().get_title(), fontsize='6')
        gfg.set_ylabel(ylabel = '', fontsize = 6)
        gfg.set_xlabel(xlabel = '', fontsize = 6)
    plt.tight_layout()
    plt.show()
