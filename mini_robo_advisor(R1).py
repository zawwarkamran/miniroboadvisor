import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as optimizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
sns.set_style('white')

print("Please input how many ETF's you would like to invest in :")
n0 = int(input())

data = pd.read_csv('sortedetf.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Filter ETF's in between the dates from 2014 to 2017
fil = ((data['Date'] >= '01-01-2012') & (data['Date'] <= '01-01-2017'))
data_timed = data.loc[fil]

# Since Some ETF's have missing data, we set a threshold to keep ETF's that
# have all the prices
data_filtered = data_timed.dropna(thresh=int(np.floor((len(data_timed))))
                                  , axis=1)
data_filtered = data_filtered.reset_index(drop=True)
data_filtered = data_filtered.drop(['Date'], axis=1)

YTD = []
for item in data_filtered:
    YTD.append(((data_filtered[item].iloc[len(data_filtered)-1]
                 - data_filtered[item].iloc[0])
                / data_filtered[item].iloc[0]))


avg_daily_percentage_change = data_filtered.pct_change().mean()
dfwithytd = pd.DataFrame((data_filtered.pct_change().std())
                         , columns=['5YearVolatility'])
dfwithytd['5YearReturn'] = YTD
dfwithytd['5YearAverageDailyReturn'] = avg_daily_percentage_change
dfwithytd.drop('soxl.us.txt', inplace=True)


Y = dfwithytd.values
# Y = scaler.fit_transform(Y)

# sns.scatterplot(x=dfwithytd['5YearReturn'], y=dfwithytd['Variance in Daily Price Change']
#                 , data=dfwithytd
#                 , s=15)
# plt.xlabel('5YearReturn')
# plt.ylabel('ADSD')
# plt.show()

kmeans = KMeans(n_clusters=n0)
kmeans.fit(Y)
K_cluster = kmeans.labels_
kmeans_dataframe = dfwithytd
kmeans_dataframe = kmeans_dataframe.reset_index()
kmeans_dataframe['Cluster'] = K_cluster


palette = sns.color_palette('bright', n0)
# sns.scatterplot(Y[:, 0], Y[:, 2], hue=kmeans.labels_, palette=palette, s=15)
# plt.title('Pre Kmeans DataFrame')
# plt.xlabel('5YearVolatility')
# plt.ylabel('5YearAverageDailyReturn')
# plt.show()
#
# sns.scatterplot(x='5YearVolatility', y='5YearAverageDailyReturn'
#                 , hue='Cluster', data=kmeans_dataframe
#                 , palette=palette, s=15)
# plt.title('Kmeans_Dataframe')
# plt.xlabel('5YearReturn')
# plt.ylabel('Variance in Daily Price Change')
# plt.show()


# Method 2: Use Hierarchical Clustering. We'll then compare the performance
# between the two:

# def rang(x):
#     if x < 2:
#         return 2
#     else:
#         return n0



nodes_lst = []
for i in range(2, n0+5):
    g = AgglomerativeClustering(n_clusters=i, affinity='euclidean'
                                , linkage='ward')
    g.fit(Y[:, [0, 2]])
    nodes_lst.append(g.labels_)

sil_score = []
for item in nodes_lst:
    sil_score.append(silhouette_score(Y[:, [0, 2]], item, metric='euclidean'))
# print(len(sil_score))
# print(sil_score)
result = np.where(sil_score == np.amax(sil_score))
lenner = int(result[0])
H_clust = nodes_lst[lenner]

hclust_dataframe = dfwithytd
hclust_dataframe = hclust_dataframe.reset_index()
hclust_dataframe['Cluster'] = H_clust


palette_2 = sns.color_palette('bright', (len(nodes_lst)))
# sns.scatterplot(Y[:, 0], Y[:, 2], hue=H_clust, palette=palette_2, s=15)
# plt.title('Pre HCLUST DATAFRAME')
# plt.xlabel('5YearReturn')
# plt.ylabel('Variance in Daily Price Change')
# plt.show()
#
# sns.scatterplot(x='5YearVolatility'
#                 , y='5YearAverageDailyReturn', hue=H_clust
#                 , data=hclust_dataframe
#                 , palette=palette_2, s=15)
# plt.title('Post HCLUST DATAFRAME')
# plt.xlabel('5YearReturn')
# plt.ylabel('Variance in Daily Price Change')
# plt.show()


# showdef volatility_min(val, etf_name, df_1, df_2):
#     assert len(val) == len(etf_name)
#     lookup = dict()
#     for i in range(len(val)):
#         lookup[etf_name[i]] = val[i]
#     mu = val.mean()
#     df_1 = df_1.replace(lookup)
#     std = sqrt(((1/len(val))*(np.sum([x-mu for x in val]))**2))
#     return std


# mu = np.sum(data_filtered['vti.us.txt'])/len(data_filtered['vti.us.txt'])
# #
# add = []
# for item in data_filtered['vti.us.txt']:
#     add.append((item-mu)**2)
#
# print(data_filtered['vti.us.txt'].std()
#       - sqrt((1/len(data_filtered['vti.us.txt']))*np.sum(add)))

cluster_weight = np.zeros(n0)
cluster_weight_2 = np.zeros(len(np.unique(nodes_lst[lenner])))

kmeans_cluster = []
for i in range(n0):
    fil = kmeans_dataframe['Cluster'] == i
    kmeans_cluster.append(kmeans_dataframe[fil])

hc_cluster = []
for i in range(len(np.unique(nodes_lst[lenner]))):
    fil = hclust_dataframe['Cluster'] == i
    hc_cluster.append(hclust_dataframe[fil])


def list_avg(lst):
    ret = []
    for h in lst:
        ret.append(h['5YearReturn'].mean())
    return ret


def risk_avg(lst):
    ret = []
    for h in lst:
        ret.append(h['5YearVolatility'].mean())
    return []


def risk_avg_2(lst):
    ret = []
    for h in lst:
        ret.append(h['5YearVolatility'].mean())
    return ret


def total_return(weights):
    return -np.sum(list_avg(kmeans_cluster)*weights)


def total_return_2(weights):
    return -np.sum(list_avg(hc_cluster)*weights)


def min_risk(weights):
    return np.sum(risk_avg(kmeans_cluster)*weights)


def min_risk_2(weights):
    return np.sum(risk_avg(hc_cluster)*weights)


bnds = tuple((0.2, 0.8) for x in range(len(cluster_weight)))
bns2 = tuple((0.2, 0.8) for x in range(len(np.unique(nodes_lst[lenner]))))

cons = {'type': 'eq', 'fun': lambda z: np.sum(z) - 1}


optim_kmeans = optimizer.minimize(total_return, x0=cluster_weight
                                  , method='trust-constr'
                                  , bounds=bnds
                                  , constraints=cons)

optim_hlcust = optimizer.minimize(total_return_2, x0=cluster_weight_2
                                  , method='trust-constr', bounds=bns2
                                  , constraints=cons)

