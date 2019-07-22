#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:55:59 2019
AlgoTrading PS3

1.	From the trade book and the order data snapshots of every minute, 
estimate the probability of a hidden order behind every displayed order 
using a classification techniques. 

2.	From the trade book and the order data snapshots of every minute, 
estimate the amount of a hidden order behind every displayed order using 
a regression techniques. 

3.	Using your answers of 1 and 2 above describe a liquidity seeking algorithm.
 
@author: Jason
"""

import numpy as np
import pandas as pd

tradebook= pd.read_excel('PS1Data_tradebook.xlsx', index_col=None, header=0);
orderbook= pd.read_excel('PS1Data_orderbook.xlsx', index_col=None, header=0);


# hidden=1 not hidden=0
hidden_series=[]
for index, row in orderbook.iterrows():
    if orderbook.loc[index, 'vd'] < orderbook.loc[index, 'vo']:
        hidden_series.append(1)
    else:
        hidden_series.append(0)
hidden=pd.DataFrame({'hidden':hidden_series})
orderbook_hidden=pd.concat([orderbook,hidden],axis=1)


#EXPORT the fill situations of each order
orderbook_hidden.to_csv('orderbook_hidden.csv',encoding='utf-8',index=False)

############################################
#Q1 METHOD 1: random forest(classification method)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

def convert(data):
    number = preprocessing.LabelEncoder()
    data['date'] = number.fit_transform(data.date)
    data['time'] = number.fit_transform(data.time)
    data['bors'] = number.fit_transform(data.bors)
    data['mkt'] = number.fit_transform(data.mkt)
    data=data.fillna(-999)
    return data

orderbook_hidden_pro=convert(orderbook_hidden)
#orderbook_hidden_pro['date'].describe()

#features
X=orderbook_hidden_pro[['date','time','bors','vd','lp','mkt']]
y=orderbook_hidden_pro['hidden']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
# 70% training and 30% test

#train a classifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#importance of ['date','time','bors','vd','lp','mkt']
print(clf.feature_importances_)

#probability matrix for 2 possibilities: not_hidden or hidden
probabilities=clf.predict_proba(X)
df_prob=pd.DataFrame(probabilities, columns=['not_hidden_prob','hidden_prob'] )
orderbook_hidden_prob=pd.concat([orderbook_hidden_pro,df_prob],axis=1)

#probabilities for each order
pd.set_option('display.expand_frame_repr', False)
orderbook_hidden_prob.head(15)
orderbook_hidden_prob['hidden_prob'].describe()


# add probabilities to the original orderbook:
orderbook= pd.read_excel('PS1Data_orderbook.xlsx', index_col=None, header=0);
new_orderbook=pd.concat([orderbook,df_prob],axis=1)
new_orderbook.to_csv('new_orderbook.csv',encoding='utf-8',index=False)

"""
Liquidity seeking algo:
    1.Using the classification method to predict whether there will be a hidden liquidity and then
    2.Predict the hidden amount using the regression method
    3.Pin down the hidden order buy submitting the order on the opposite side of the hidden order predicted
    4.If there isn't one as predicted, cancel the order immediately

Potential Problem:
    1.The linear regression I used for the hidden amount prediction isn't accurate enough since it's linear
    and used for all orders

Potential improvement: 
    1: filter out all the hidden orders and then run the linear regression
    2: Use classification to predict hidden or not and then use the linear model trained in step one to 
    predict the amount
"""