#!/usr/bin/env python
# coding: utf-8

# # Contributors:
# 1) Shrey Shah: https://github.com/shreycshah
# 2) Tannavi Snehal: https://github.com/Tannavi-Snehal
# 3) Gautam Singh: https://github.com/ggs597

# Go through README.md for the overview of the hackathon

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# # Reading and analysing dataset

# In[2]:


data=pd.read_csv(r'Train.csv')
test=pd.read_csv(r'Test.csv')
test.drop(columns=['price'],axis=1,inplace=True)
data.shape,test.shape


# In[3]:


data.isnull().sum()


# In[4]:


test.isnull().sum()


# # Handling 'availability' column

# In[5]:


data.availability.unique()


# In[6]:


data['availability']=np.where(data['availability'].str.contains('Ready To Move'),str(13),
                     np.where( data['availability'].str.contains('Immediate Possession'),str(14),
                        data.availability.apply(lambda x:x[3:])
                        ))
data.head()

test['availability']=np.where(test['availability'].str.contains('Ready To Move'),str(13),
                     np.where( test['availability'].str.contains('Immediate Possession'),str(14),
                        test.availability.apply(lambda x:x[3:])
                        ))

print(data['availability'].unique())

#later on we will encode the availability column
data.head()


# # Handling 'size' columns

# In[7]:


data['size'].unique()


# In[8]:


data['size']=data['size'].fillna(data['size'].mode()[0])
data['size']=data['size'].str.extract('(\d+)')
data['size']=data['size'].astype('float')

test['size']=test['size'].fillna('2.0')
test['size']=test['size'].str.extract('(\d+)')
test['size']=test['size'].astype('float')

data.head()


# # Handling 'bath', 'balcony' & 'location' columns

# In[9]:


data['bath']=data['bath'].fillna(data['bath'].median())
data['balcony']=data['balcony'].fillna(data['balcony'].median())
data['location']=data['location'].fillna(data['location'].mode()[0])

test['bath']=test['bath'].fillna(data['bath'].median())
test['balcony']=test['balcony'].fillna(data['balcony'].median())
test['location']=test['location'].fillna(data['location'].mode()[0])


# # Handling 'total_sqft' column

# In[10]:


import re
def preprocess_total_sqft(my_list):
    if len(my_list) == 1:
        
        try:
            return float(my_list[0])
        except:
            strings = ['Sq. Meter', 'Sq. Yards', 'Perch', 'Acres', 'Cents', 'Guntha', 'Grounds']
            split_list = re.split('(\d*.*\d)', my_list[0])[1:]
            area = float(split_list[0])
            type_of_area = split_list[1]
            
            if type_of_area == 'Sq. Meter':
                area_in_sqft = area * 10.7639
            elif type_of_area == 'Sq. Yards':
                area_in_sqft = area * 9.0
            elif type_of_area == 'Perch':
                area_in_sqft = area * 272.25
            elif type_of_area == 'Acres':
                area_in_sqft = area * 43560.0
            elif type_of_area == 'Cents':
                area_in_sqft = area * 435.61545
            elif type_of_area == 'Guntha':
                area_in_sqft = area * 1089.0
            elif type_of_area == 'Grounds':
                area_in_sqft = area * 2400.0
            return float(area_in_sqft)
        
    else:
        return (float(my_list[0]) + float(my_list[1]))/2.0

data['total_sqft'] = data.total_sqft.str.split('-').apply(preprocess_total_sqft)
test['total_sqft'] = test.total_sqft.str.split('-').apply(preprocess_total_sqft)


# # Target Encoding for categorical columns

# In[11]:


area_type_Encoding=data.groupby('area_type')['price'].mean()
data['area_type']=data['area_type'].map(area_type_Encoding)
test['area_type']=test['area_type'].map(area_type_Encoding)

location_Encoding=data.groupby('location')['price'].mean()
data['location']=data['location'].map(location_Encoding)
test['location']=test['location'].map(location_Encoding)

availability_Encoding=data.groupby('availability')['price'].mean()
data['availability']=data['availability'].map(availability_Encoding)
test['availability']=test['availability'].map(availability_Encoding)

data.head()


# # Outlier Handling in price and total_sqft using z-score

# In[12]:


sns.boxplot(data['price'])


# In[13]:


sns.boxplot(data['total_sqft'])


# In[14]:


from scipy import stats
data['z_score_price'] = np.abs(stats.zscore(data['price']))
data['z_score_sqft']=np.abs(stats.zscore(data['total_sqft']))
data=data[data['z_score_price']<=3]
data=data[data['z_score_sqft']<=3]
data.drop(columns=['z_score_price','z_score_sqft'],axis=1,inplace=True)
data.shape


# # Feature Engineering

# In[15]:


data['price']=data['price']*100000


# In[16]:


lookup=data.groupby(['location','society'])['price'].agg(['mean']).reset_index()
lookup=lookup.rename(columns={'mean':'location_society_mean'})
data=pd.merge(data,lookup,how='left',left_on=['location','society'],right_on=['location','society'])
test=pd.merge(test,lookup,how='left',left_on=['location','society'],right_on=['location','society'])
lookup


# In[17]:


lookup1=data.groupby(['society','size'])['price'].agg(['mean']).reset_index()
lookup1=lookup1.rename(columns={'mean':'society_size_mean'})
data=pd.merge(data,lookup1,how='left',left_on=['size','society'],right_on=['size','society'])
test=pd.merge(test,lookup1,how='left',left_on=['size','society'],right_on=['size','society'])
lookup1


# In[18]:


lookup2=data.groupby(['area_type','size'])['price'].agg(['mean']).reset_index()
lookup2=lookup2.rename(columns={'mean':'area_size_mean'})
data=pd.merge(data,lookup2,how='left',left_on=['area_type','size'],right_on=['area_type','size'])
test=pd.merge(test,lookup2,how='left',left_on=['area_type','size'],right_on=['area_type','size'])
lookup2


# In[19]:


lookup3=data.groupby(['bath','balcony'])['price'].agg(['mean']).reset_index()
lookup3=lookup3.rename(columns={'mean':'bath_balcony_mean'})
data=pd.merge(data,lookup3,how='left',left_on=['bath','balcony'],right_on=['bath','balcony'])
test=pd.merge(test,lookup3,how='left',left_on=['bath','balcony'],right_on=['bath','balcony'])
lookup3


# In[20]:


lookup4=data.groupby(['bath','size'])['price'].agg(['mean']).reset_index()
lookup4=lookup4.rename(columns={'mean':'bath_size_mean'})
data=pd.merge(data,lookup4,how='left',left_on=['bath','size'],right_on=['bath','size'])
test=pd.merge(test,lookup4,how='left',left_on=['bath','size'],right_on=['bath','size'])
lookup4


# In[21]:


lookup5=data.groupby(['size','balcony'])['price'].agg(['mean']).reset_index()
lookup5=lookup5.rename(columns={'mean':'size_balcony_mean'})
data=pd.merge(data,lookup5,how='left',left_on=['size','balcony'],right_on=['size','balcony'])
test=pd.merge(test,lookup5,how='left',left_on=['size','balcony'],right_on=['size','balcony'])
lookup5


# In[22]:


lookup6=data.groupby(['availability','society'])['price'].agg(['mean']).reset_index()
lookup6=lookup6.rename(columns={'mean':'availability_society_mean'})
data=pd.merge(data,lookup6,how='left',left_on=['availability','society'],right_on=['availability','society'])
test=pd.merge(test,lookup6,how='left',left_on=['availability','society'],right_on=['availability','society'])
lookup6


# In[23]:


lookup7=data.groupby(['area_type','society'])['price'].agg(['mean']).reset_index()
lookup7=lookup7.rename(columns={'mean':'area_society_mean'})
data=pd.merge(data,lookup7,how='left',left_on=['area_type','society'],right_on=['area_type','society'])
test=pd.merge(test,lookup7,how='left',left_on=['area_type','society'],right_on=['area_type','society'])
lookup7


# In[24]:


data['price_per_sqft']=data['price']/data['total_sqft']
price_per_sqft_area_wise=data.groupby('area_type')['price_per_sqft'].mean()
data['price_per_sqft_area_wise']=data['area_type'].map(price_per_sqft_area_wise)
data.drop(['price_per_sqft'],axis=1,inplace=True)
test['price_per_sqft_area_wise']=test['area_type'].map(price_per_sqft_area_wise)


# In[25]:


data['price']=data['price']/100000


# In[26]:


data.drop(columns=['society'],axis=1,inplace=True)
test.drop(columns=['society'],axis=1,inplace=True)


# In[27]:


data.head()


# # Train Validation Split

# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(
    data.drop(columns=['price']), data['price'], test_size=0.2, random_state=2)

X_train.shape, X_validation.shape


# # Feature Selection

# In[29]:


features=['area_type','availability', 'location', 'size', 'total_sqft', 'bath', 'balcony', 'society_size_mean', 
           'area_size_mean', 'price_per_sqft_area_wise','bath_balcony_mean','bath_size_mean',
          'size_balcony_mean','availability_society_mean','area_society_mean']


# # Training Model

# In[30]:


from xgboost import XGBRegressor
xgb = XGBRegressor(random_state=2)
xgb.fit(X_train[features],y_train)
y_pred_xgb=xgb.predict(X_validation[features])
score=1-np.sqrt(np.square(np.log(y_pred_xgb +1) - np.log(y_validation +1)).mean()) 
score


# In[31]:


from lightgbm import LGBMRegressor
lgb = LGBMRegressor(random_state=2)
lgb.fit(X_train[features],y_train)
y_pred_lgb=lgb.predict(X_validation[features])
score=1-np.sqrt(np.square(np.log(y_pred_lgb +1) - np.log(y_validation +1)).mean()) 
score


# In[32]:


from catboost import CatBoostRegressor
cat = CatBoostRegressor(random_state=100,logging_level='Silent')
cat.fit(X_train[features],y_train)
y_pred_cat=cat.predict(X_validation[features])
score=1-np.sqrt(np.square(np.log(y_pred_cat +1) - np.log(y_validation +1)).mean()) 
score


# In[33]:


from sklearn.ensemble import VotingRegressor

xgb = XGBRegressor(random_state=6,learning_rate=0.025, n_estimators=500,max_depth=8)
lgb =  LGBMRegressor(boosting_type='dart', learning_rate=0.2, max_depth=6,n_estimators=350, num_leaves=45, random_state=0)
cat = CatBoostRegressor(random_state=100,logging_level='Silent',l2_leaf_reg=2,learning_rate=0.01,iterations=1200,depth=6)

vr = VotingRegressor([('xgb', xgb), ('lgb', lgb),('cat', cat)])

y_pred_vr=vr.fit(X_train[features], y_train).predict(X_validation[features])

score=1-np.sqrt(np.square(np.log(y_pred_vr +1) - np.log(y_validation +1)).mean()) 
score


# # Final Prediction

# In[34]:


y_final_test=vr.predict(test[features])


# In[35]:


subm=pd.DataFrame({'price':y_final_test})
subm[subm['price']<0]


# In[36]:


#subm.to_csv('best.csv',index=False) 


# In[37]:


# import pickle
# pickle.dump(vr,open('model.pkl','wb'))

