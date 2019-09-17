import os
import pandas as pd
import tensorflow as tf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.model_selection import train_test_split
os.chdir("/Users/zawwark/Documents/PycharmProjects/ML/ETF ROBO STUFF/ETFs")


data_lst = []
for i in os.listdir('/Users/zawwark/Documents/PycharmProjects/ML/ETF ROBO STUFF/ETFs'):
    if i.endswith('.us.txt'):
        g = pd.read_csv(i, sep=",")
        g["ETF Name"] = i
        g["Date"] = pd.to_datetime(g['Date'])
        data_lst.append(g)


data_timed = []
for item in data_lst:
    fil = item['Date'] > '01-01-2015'
    item = item.loc[fil]
    data_timed.append(item)

df = data_lst[0]
df = df.drop('Date', axis=1)
df = df.drop('ETF Name', axis=1)
df = df.drop('OpenInt', axis=1)
df_close = df['Close']
df = df.drop('Close', axis=1)
print(df.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)

df_scaled_transformed = scaler.fit_transform(df)

df = pd.DataFrame(df_scaled_transformed, columns=df.columns)

X = df
y = df_close

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,
                                                    shuffle=False)


n_variables = 4

weight_initializer = tf.compat.v1.variance_scaling_initializer(mode='fan_avg')

X = tf.compat.v1.placeholder(tf.int64, shape=[None, n_variables])
Y = tf.compat.v1.placeholder(tf.int64, shape=[None])


n_neurons_1 = 8
n_neurons_2 = 4
n_neurons_3 = 2
n_targets = 1

hidden_1 = tf.compat.v1.Variable(weight_initializer)







