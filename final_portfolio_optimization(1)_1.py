#!/usr/bin/env python
# coding: utf-8

# # Install the library

# In[ ]:


pip install yfinance


# In[ ]:


pip install PyPortfolioOpt


# # Data Preprocessing:

# # Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from pypfopt.discrete_allocation import DiscreteAllocation
from scipy.cluster.hierarchy import dendrogram, linkage


# # Define a list of stock symbols

# In[ ]:


symbols = ['RELIANCE.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','ITC.NS','TCS.NS','KOTAKBANK.NS','LT.NS','AXISBANK.NS','ADANIPORTS.NS','ASIANPAINT.NS','BAJAJ-AUTO.NS','BAJFINANCE.NS','BAJAJFINSV.NS','BHARTIARTL.NS','BRITANNIA.NS','CIPLA.NS','COALINDIA.NS','DIVISLAB.NS','DRREDDY.NS','EICHERMOT.NS','GRASIM.NS','HCLTECH.NS','HEROMOTOCO.NS','HINDALCO.NS','HINDUNILVR.NS','IOC.NS','INDUSINDBK.NS','JSWSTEEL.NS','M&M.NS','MARUTI.NS','NTPC.NS','NESTLEIND.NS','ONGC.NS','POWERGRID.NS','SHREECEM.NS','SBIN.NS','SUNPHARMA.NS','TATAMOTORS.NS','TATASTEEL.NS','TECHM.NS','TITAN.NS','ULTRACEMCO.NS','UPL.NS','WIPRO.NS','ADANIGREEN.NS','ADANIENT.NS','HINDPETRO.NS','BPCL.NS']


# # Download stock data using Yahoo Finance API

# In[ ]:


num_companies = len(symbols)
weight= 1 / num_companies
weight= np.full(num_companies,weight)
for i, company in enumerate(symbols):
    print(f"{company}: {weight[i]:.4f}")


# # Define the start and end dates for data retrieval

# In[ ]:


start_date = '2022-01-01'
end_date =  datetime.today().strftime('%Y-%m-%d')


# # Determine each stock's ideal weight

# In[ ]:


data = yf.download(symbols, start=start_date, end=end_date)["Adj Close"]


# In[ ]:


data


# # Check for missing values in the data

# In[ ]:


data.isnull().sum()


# # Generate descriptive statistics for the 'Adj Close' data

# In[ ]:


data.describe()


# # Calculate the daily returns based on the 'Adj Close' prices

# In[ ]:


returns= data.pct_change()
returns


# In[ ]:


returns.dropna(axis=0, inplace=True)
returns


# # EDA

# # Calculate descriptive statistics of the 'Adj Close' prices

# In[ ]:


data.describe()


# # Visualize the distribution of returns

# In[ ]:


returns.hist(bins=40, figsize=(40, 24))
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Distribution of Returns')


# # Calculate and plot the correlation matrix

# In[ ]:


corr_matrix = returns.corr()
plt.figure(figsize=(30,25))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", annot_kws={"size":10})
plt.xticks(range(len(symbols)), symbols, rotation='vertical', fontsize=10)
plt.yticks(range(len(symbols)), symbols, fontsize=18)
plt.title('Correlation Matrix of Returns', fontsize=30)
plt.show()


# # Plotting Adj Close Price

# In[ ]:


title = 'Portfolio Adj Close Price'
my_stock =data
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=my_stock)
plt.title(title, fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.ylabel("Adj Close", fontsize=18)
plt.legend(labels=my_stock.columns, loc="upper left")
plt.tight_layout()
plt.show()


# # Annualized covariance matrix

# In[ ]:


covariance_matrix=returns.cov() * 252
covariance_matrix


# # Calculate Portfolio Variance

# In[ ]:


port_variance= np.dot(weight, np.dot(covariance_matrix, weight))
print("Portfolio Variance:", port_variance)


# # Calculate Volatility for Portfolio

# In[ ]:


port_volatility= np.sqrt(port_variance)
port_volatility


# In[ ]:


portfolio_annual_return= np.sum(returns.mean() * weight) * 252
portfolio_annual_return


# In[ ]:


percent_var= str(round(port_variance,2)*100) + '%'
percent_vols= str(round(port_volatility,2)*100) + '%'
percent_ret= str(round(portfolio_annual_return,2)*100) + '%'

print('Expected annual return:' +percent_ret)
print('Annual volatility/risk :' +percent_vols)
print('Annual variance :' +percent_var)


# # calculate the expected returns and annualised sample covariance matrix of asset returns

# In[ ]:


mu = expected_returns.mean_historical_return(data)
cov_matrix = risk_models.sample_cov(data)
ef = EfficientFrontier(mu, cov_matrix)


# # Optimize for max sharpe

# In[ ]:


weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)


# # Discrete allocation of each stock

# In[ ]:


latest_prices = data.iloc[-1]
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()

print("Discrete Allocation:")
for stock, shares in allocation.items():
    print(f"{stock}: {shares} shares")


# # Performing Hierarchical Clustering

# In[ ]:


num_clusters = 3
hierarchical_model = AgglomerativeClustering(n_clusters=num_clusters)
clusters = hierarchical_model.fit_predict(returns)

for i, symbol in enumerate(symbols):
    print(f"{symbol}: Cluster {clusters[i]}")


# # Calculating Linkage Matrix

# In[ ]:


linkage_matrix = linkage(returns, method='complete')
print("Number of symbols:", len(symbols))
print("Number of rows in linkage matrix:", linkage_matrix.shape[0])

print(linkage_matrix[:10])

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Merged Clusters')
plt.ylabel('Distance')
plt.show()


# # Perform clustering of portfolios using K-means and calculating silhouette_score

# In[ ]:


kmeans_model = KMeans(n_clusters=num_clusters)
kmeans_clusters = kmeans_model.fit_predict(returns)

silhouette_avg = silhouette_score(returns, kmeans_clusters)
sample_silhouette_values = silhouette_samples(returns, kmeans_clusters)


# # Visualising Parallel plot using K-means Clustering

# In[ ]:


plt.figure(figsize=(10, 6))
parallel_plot_data = pd.DataFrame(data=returns, columns=symbols)
parallel_plot_data['Cluster'] = kmeans_clusters
sns.set_palette("Set1",n_colors=num_clusters)
pd.plotting.parallel_coordinates(parallel_plot_data, 'Cluster')
plt.title('Parallel Coordinate Plot for K-means Clusters')
plt.xlabel('Stock Symbols')
plt.ylabel('Returns')
plt.show()


# # Plotting using silhouette for k-means Clustering

# In[ ]:


plt.figure(figsize=(8, 6))
y_lower = 10
for i in range(num_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[kmeans_clusters == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=f'C{i}', alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.title("Silhouette Plot for K-means Clustering")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.yticks([])
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()


# # Performing PCA and Visualising

# In[ ]:


pca = PCA(n_components=2)
pca_components = pca.fit_transform(returns)


# In[ ]:


pca_df = pd.DataFrame(data=pca_components, columns=['Principal Component 1', 'Principal Component 2'])
sns.set(style="ticks")
sns.pairplot(pca_df)
plt.suptitle("Scatter Plot Matrix for PCA Components", y=1.02)
plt.show()


# In[ ]:


loadings_df = pd.DataFrame(pca.components_, columns=symbols)
plt.figure(figsize=(12, 8))
loadings_df.plot(kind='bar')
plt.title('Feature Loadings for PCA Components')
plt.xlabel('Principal Components')
plt.ylabel('Feature Loadings')
plt.grid(True)
plt.show()

