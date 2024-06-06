#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy.stats import skew
import scipy.stats 
from scipy.special import boxcox1p
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)
from sklearn.preprocessing import LabelEncoder


# In[2]:


df1 = pd.DataFrame({'type': ['Apartment', 'Studio Apartment', 'Villa', 'Independent House', 'Penthouse']})

# Convert categorical data to numerical data using cat.codes
df1['type'] = df1['type'].astype('category')
df1['type_Codes'] = df1['type'].cat.codes

# View the converted DataFrame
print(df1)


# In[3]:


df1 = pd.read_csv(r"C:\Users\adala\OneDrive\Desktop\PriceWise\Mumbai House Prices.csv")
df1.head()


# In[4]:


df1.shape


# In[5]:


df1.describe()


# In[6]:


df1['region'].value_counts()


# In[7]:


df1['type'].value_counts()


# In[8]:


df1['age'].value_counts()


# In[9]:


df1.isnull().sum()


# In[10]:


df1['status'].unique()


# In[51]:


df2 = df1.drop(['age', 'locality', 'status'], axis=1)
df2.head()


# In[12]:





# In[13]:


df2['bhk'].unique()


# In[59]:


conversion_factor_crore_to_lakh = 100  # 1 Crore = 100 Lakhs

def convert_to_lakhs(row):
    if row["price_unit"] == "L":
        return row["price"]
    elif row["price_unit"] == "Cr":
        return row["price"] * conversion_factor_crore_to_lakh

df2["price_lakhs"] = df2.apply(convert_to_lakhs, axis=1)

df3 = df2.drop(['price','type', 'price_unit'], axis=1)
df3.head()


# In[45]:


ax = sns.distplot(df3['price_lakhs'])


# In[46]:


log_transform_y =  np.log(df3['price_lakhs'])


# In[17]:


ax = sns.distplot(log_transform_y)


# In[47]:


df4 = df3.copy()
df4['price_per_sqft'] = df4['price_lakhs']/df4['area']
df4.head()


# In[48]:


df4['price_per_sqft'].describe()


# In[28]:


numerical_features = df4[['bhk', 'area', 'price_lakhs', 'price_per_sqft']]
categorical_features = df4.select_dtypes(include=['object'])


# In[55]:


scaler = StandardScaler()

standardized_features = scaler.fit_transform(numerical_features)

df4 = pd.DataFrame(standardized_features, columns=numerical_features.columns)
df4.head()


# In[49]:


# Exclude non-numeric columns and compute the correlation matrix
numeric_df4 = df4.select_dtypes(include=[np.number])
correlation_matrix = numeric_df4.corr()
print(df4.head())
print(df4.dtypes)


# In[56]:


df4.corr()


# In[33]:


for feature in numerical_features.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df4, x=feature, y=log_transform_y)
    plt.title(f'Scatter Plot of {feature} vs price_lakhs')
    plt.xlabel(feature)
    plt.ylabel('Target Variable')
    plt.show()


# In[60]:


for feature in numerical_features.columns:
    sns.residplot(data=df4, x=feature, y=log_transform_y)
    plt.title(f'Residual Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Residuals')
    plt.show()


# In[61]:


skewed_feats = numerical_features.apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_feats


# In[62]:


df_fix_skew = np.log1p(numerical_features)


# In[63]:


skewed_feats = df_fix_skew.apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_feats


# In[64]:


for i, column in enumerate(df_fix_skew.columns):
    plt.figure(figsize=(6, 4))
    plt.boxplot(df4[column], vert=False, boxprops=dict())
    plt.title(f'Box Plot of {column}')
    plt.xlabel('Value')
    plt.show()


# In[65]:


scaler = StandardScaler()

standardized_features = scaler.fit_transform(df_fix_skew)

std_df4 = pd.DataFrame(standardized_features, columns=numerical_features.columns)
std_df4.head()


# In[ ]:


std_df4.corr()


# In[66]:


df5 = pd.concat([df4, categorical_features], axis=1)
df5.head()


# In[68]:


df5.region = df5.region.apply(lambda x: x.strip())
location_stats = df5['region'].value_counts(ascending=False)
location_stats


# In[69]:


location_stats.values.sum()


# In[70]:


len(location_stats[location_stats>10])


# In[71]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[72]:


len(df5.region.unique())


# In[73]:


df5.region = df5.region.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df4.region.unique())


# In[74]:


df5.region.value_counts()


# In[76]:


target_col = 'price_lakhs'
df5 = df5[[col for col in df5.columns if col != target_col] + [target_col]]
df5.head()


# In[77]:


df5 = df5.copy()
df5 = pd.get_dummies(df5, columns=['type'], prefix=['type'])
df5.head()


# In[78]:


df_encoded = pd.get_dummies(df5, columns=['region'], prefix=['region'])


# In[79]:


df_encoded.head()


# In[80]:


df_encoded.shape


# In[83]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[87]:


X = df_encoded.drop('price_lakhs', axis=1)
y = log_transform_y


# In[88]:


X


# In[85]:


# Add a constant to the X_scaled matrix for intercept term
X_scaled = sm.add_constant(X)

model = sm.OLS(y, X_scaled).fit()

print(model.summary())


# In[ ]:




