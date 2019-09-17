import os
import pandas as pd
from functools import reduce
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
os.chdir("directory with the data")


data_lst = []
for i in os.listdir('directory with the data'):
    if i.endswith('.us.txt'):
        g = pd.read_csv(i, sep=",")
        for name in g.columns:
            if name != 'Close' and name != 'Date':
                g = g.drop(name, axis=1)
        g['Date'] = pd.to_datetime(g['Date'])
        g = g.rename(columns={'Close': i})
        data_lst.append(g)

os.chdir('directory where you want to save the merged data')

df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'],
                                                how='outer'),
                   data_lst).fillna('void')

df_merged.to_csv('merged.txt', sep=',', na_rep='.', index=False)
