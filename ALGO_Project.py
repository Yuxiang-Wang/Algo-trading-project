#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:41:35 2019

@author: Jason
"""
import pandas as pd
import datetime as dt
import numpy as np
orderbook= pd.read_excel('PS1Data_orderbook.xlsx', index_col=None, header=0);
df_orderbook = orderbook[orderbook['date'] == '2006-04-03']


best_bid = []
best_offer = []
trade_list = []

def add_order(order, vol=0):
    temp = {}
    temp['date'] = order['date']
    temp['time'] = order['time']
    temp['orno'] = order['orno']
    if vol == 0:
        temp['vd'] = order['vd']
        temp['vo'] = order['vo']
    else:
        temp['vd'] = vol
        temp['vo'] = vol        
    temp['lp'] = order['lp']
    return temp

     
def check_orno(order): 
    global best_bid
    global best_offer
    if order['bors'] == 'B':
        temp_orders = [x['orno'] for x in best_bid]
        if order['orno'] in temp_orders:
            i = temp_orders.index(order['orno'])
            if order['vo'] == best_bid[i]['vo'] and order['vd'] == best_bid[i]['vd'] and order['lp'] == best_bid[i]['lp']:
                del best_bid[i]
                return False
            else:
                del best_bid[i]
                return True
        else:
            return True
    else:
        temp_orders = [x['orno'] for x in best_offer]
        if order['orno'] in temp_orders:
            i = temp_orders.index(order['orno'])
            if order['vo'] == best_offer[i]['vo'] and order['vd'] == best_offer[i]['vd'] and order['lp'] == best_offer[i]['lp']:
                del best_offer[i]
                return False
            else:
                del best_offer[i]
                return True
        else:
            return True          

def construct_tradebook(index, order, vol_trade):
    if order['bors'] == 'B':
        #print(index,'trade occured at', best_offer[0]['lp'], vol_trade, order['orno'], best_offer[0]['orno'])
        temp_dict = {'symbol':order['symbol'], 'series': order['series'], 'date': order['date'].date(), 'time': order['time'],
                    'lp':best_offer[0]['lp'], 'vol': vol_trade, 'bno': order['orno'],
                    'sno':best_offer[0]['orno']}
    else:
        #print(index, 'trade occured at', best_bid[0]['lp'], vol_trade, best_bid[0]['orno'], order['orno'])
        temp_dict = {'symbol':order['symbol'], 'series': order['series'], 'date': order['date'].date(), 'time': order['time'],
            'lp':best_bid[0]['lp'], 'vol': vol_trade, 'bno': best_bid[0]['orno'],
            'sno':order['orno']}
    trade_list.append(temp_dict)
            

def matcher(n):
    global best_bid
    global best_offer
    for index, order in df_orderbook.iloc[0:n,].iterrows():
        if check_orno(order):
            vol = order['vo']

            #buy orders
            if order['bors'] == 'B':
                if len(best_offer) == 0:
                    best_bid.append(add_order(order))
                    best_bid = sorted(best_bid, key = lambda x: x['lp'], reverse = True)
                else:
                    if order['mkt'] == 'Y':
                        while vol > 0:
                            if vol < best_offer[0]['vo']:
                                vol_trade = vol
                                best_offer[0]['vo'] -= vol
                                construct_tradebook(index, order, vol_trade)
                                vol = 0
                            else:
                                vol_trade = best_offer[0]['vo']
                                vol = vol - best_offer[0]['vo']
                                construct_tradebook(index, order, vol_trade)
                                best_offer[0]['vo'] =0
                                del best_offer[0]
                            #check_zero()
                    else:
                        if order['lp'] < best_offer[0]['lp']:
                            best_bid.append(add_order(order))
                            best_bid = sorted(best_bid, key = lambda x: x['lp'], reverse = True)
                        else:
                            while vol > 0 and order['lp'] >= best_offer[0]['lp'] and len(best_offer) != 0:
                                if vol < best_offer[0]['vo']:
                                    vol_trade = vol
                                    best_offer[0]['vo'] -= vol
                                    construct_tradebook(index, order, vol_trade)
                                    vol = 0
                                else:
                                    vol_trade = best_offer[0]['vo']
                                    vol = vol - best_offer[0]['vo']
                                    construct_tradebook(index, order, vol_trade)
                                    best_offer[0]['vo'] = 0
                                    del best_offer[0]
                                #check_zero()
                            if vol > 0:
                                best_bid.append(add_order(order, vol))
                                best_bid = sorted(best_bid, key = lambda x: x['lp'], reverse = True)

            #sell orders                   
            elif order['bors'] == 'S':
                if len(best_bid) == 0:
                    best_offer.append(add_order(order))
                    best_offer = sorted(best_offer, key = lambda x: x['lp'])
                else:
                    if order['mkt'] == 'Y':
                        while vol > 0:
                            if vol < best_bid[0]['vo']:
                                vol_trade = vol
                                construct_tradebook(index, order, vol)
                                best_bid[0]['vo'] -= vol
                                vol = 0
                            else:
                                vol_trade = best_bid[0]['vo']
                                vol = vol - best_bid[0]['vo']
                                construct_tradebook(index, order, vol)
                                best_bid[0]['vo'] =0
                                del best_bid[0]
                            #check_zero()
                    else:
                        if order['lp'] > best_bid[0]['lp']:
                            best_offer.append(add_order(order))
                            best_offer = sorted(best_offer, key = lambda x: x['lp'])
                        else:
                            while vol > 0 and order['lp'] <= best_bid[0]['lp'] and len(best_bid) != 0:
                                if vol < best_bid[0]['vo']:
                                    vol_trade = vol
                                    construct_tradebook(index, order, vol_trade)
                                    best_bid[0]['vo'] -= vol
                                    vol = 0
                                else:
                                    vol_trade = best_bid[0]['vo']
                                    vol = vol - best_bid[0]['vo']
                                    construct_tradebook(index, order, vol_trade)
                                    best_bid[0]['vo'] = 0
                                    del best_bid[0]
                                #check_zero()
                            if vol > 0:
                                best_offer.append(add_order(order, vol))
                                best_offer = sorted(best_offer, key = lambda x: x['lp'])

            best_bid = sorted(best_bid, key = lambda x: x['time'])
            best_bid = sorted(best_bid, key = lambda x: x['lp'], reverse = True)
            best_offer = sorted(best_offer, key = lambda x: x['time'])
            best_offer = sorted(best_offer, key = lambda x: x['lp'])
              

try:
    #after 15:30 there is a discontinuity jump to 15:50 with a lot of market orders. do until 15:30
    matcher(16396)
except:
    print("An exception occurred")

df_tradebook = pd.DataFrame(trade_list)
df_tradebook = df_tradebook[['symbol', 'series', 'date', 'time', 'lp', 'vol', 'bno', 'sno']]
pd.set_option('display.max_columns', None)
df_tradebook.head(20)

best_bid = sorted(best_bid, key = lambda x: x['time'])
best_bid_df= pd.DataFrame(best_bid);
best_bid_df.head(10)

best_offer = sorted(best_offer, key = lambda x: x['time'])
best_offer_df= pd.DataFrame(best_offer);
best_offer_df.head(10)



import pandas as pd
import datetime as dt
import numpy as np


tradebook= pd.read_excel('PS1Data_tradebook.xlsx', index_col=None, header=0);
orderbook= pd.read_excel('PS1Data_orderbook.xlsx', index_col=None, header=0);

tradebook0403 = tradebook[tradebook['date'] == '2006-04-03']

price_volume= [['time','volume','price*volume','last_traded_price']]
for i in range(len(tradebook0403)):
    d = tradebook0403.iloc[i,2]
    t = tradebook0403.iloc[i,3]
    time = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=t.hour,minute =t.minute,second =0)
    templist1 = [time, tradebook0403.iloc[i,6],tradebook0403.iloc[i,5]*tradebook0403.iloc[i,6],tradebook0403.iloc[i,5]]
    price_volume.append(templist1)
price_volume = pd.DataFrame(price_volume[1:],columns = price_volume[0])

#bench mark: VWAP minute by minute for all trades
VWAP_minute = pd.pivot_table(price_volume, values=['volume', 'price*volume','last_traded_price'], index='time',aggfunc={'volume':np.sum,'price*volume':np.sum,'last_traded_price':np.mean})
VWAP_minute['cum_sum_price*volume'] = VWAP_minute['price*volume'].cumsum()
VWAP_minute['cum_sum_volume'] = VWAP_minute['volume'].cumsum()
VWAP_minute['VWAP'] = (VWAP_minute['cum_sum_price*volume']/VWAP_minute['cum_sum_volume']).round(2)
TradeVWAP = VWAP_minute[['volume','VWAP','last_traded_price']]
TradeVWAP

best_bid_by_minute=[['time','bblp','bbv']]
for i in range(len(best_bid_df)):
    d=best_bid_df.iloc[i,0]
    t=best_bid_df.iloc[i,3]
    time = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=t.hour,minute =t.minute,second =0)
    templist=[time, best_bid_df.iloc[i,1],best_bid_df.iloc[i,5]]
    best_bid_by_minute.append(templist)
best_bid_by_minute=pd.DataFrame(best_bid_by_minute[1:],columns=best_bid_by_minute[0]) 
best_bid_by_minute=pd.pivot_table(best_bid_by_minute,values=['bblp','bbv'],index=['time'],aggfunc={'bblp':np.mean,'bbv':np.sum})

best_offer_by_minute=[['time','bolp','bov']]
for i in range(len(best_offer_df)):
    d=best_offer_df.iloc[i,0]
    t=best_offer_df.iloc[i,3]
    time = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=t.hour,minute =t.minute,second =0)
    templist=[time, best_offer_df.iloc[i,1],best_offer_df.iloc[i,5]]
    best_offer_by_minute.append(templist)
best_offer_by_minute=pd.DataFrame(best_offer_by_minute[1:],columns=best_offer_by_minute[0]) 
best_offer_by_minute=pd.pivot_table(best_offer_by_minute,values=['bolp','bov'],index=['time'],aggfunc={'bolp':np.mean,'bov':np.sum})

frames=[best_offer_by_minute,best_bid_by_minute,TradeVWAP]
result=pd.concat(frames,axis=1, sort=False)

#NBBO and corresponding volume
result.fillna(method='ffill',inplace=True)
result.iloc[0]
result.index[0]
result_x=result.iloc[2:,:]

#draw a graph containing time series of NBBO, last traded price and VWAP
import matplotlib.pyplot as plt
result_x['5MA']=result_x['last_traded_price'].rolling(window=5).mean()
result_x['10MA']=result_x['last_traded_price'].rolling(window=10).mean()

result_x.plot(use_index=True, y=['bblp', 'bolp','VWAP','last_traded_price','5MA','10MA'], figsize=(20,20), grid=True)




### Task: BUYING 10,000 SHARES OF STOCK

###1: SIMPLE VWAP ALGO
###### Description and Assumption:
### take liquidity whenever last traded price is below VWAP, minute by minute
### only take the best offer of the minute
### assuming no additional price impact to the market since only taking the liquidity of best offer
### same assumption for the other 2 algos below
### cost analysis

buy_amount=100
buy_remaining1=100
crossed_orders1=[]
decision_price=result_x.iloc[0,6]
#decision_price=result_x.iloc[0,]
for i in range(len(result_x)):
    VWAP= result_x.iloc[i,5]
    price= result_x.iloc[i,6]
    offer_amount= result_x.iloc[i,1]
    if price < VWAP and buy_remaining1 > 0:
        if offer_amount<=buy_remaining1:
            buy_remaining1=buy_remaining1-offer_amount
            crossed_orders1.append(result_x.iloc[i,:])
        else:
            buy_remaining1=0
            crossed_orders1.append(result_x.iloc[i,:])
    elif buy_remaining1==0:
        print('orders all filled')
        break
#convert list to dataframe
crossed_orders1_df=pd.DataFrame(crossed_orders1)
#adjust the last traded amount
crossed_orders1_df.iloc[-1,1]=buy_amount-sum(crossed_orders1_df.iloc[:-1,1])
#implementation shortfall calculation
is1=0
for i in range(len(crossed_orders1_df)):
    is1+=crossed_orders1_df.iloc[i,1]*(crossed_orders1_df.iloc[i,6]-decision_price)
is1_pct=is1/(100*decision_price)
#IS is about 11.9 bp, considering trading cost: 12.9bp

#Reversion Analysis: from decision to last order and 5, 10, 15, 20 minutes after the order:
a1=decision_price
b1=crossed_orders1_df.iloc[-1,6]

c1=result_x.iloc[22,6]
d1=result_x.iloc[22+5,6]
e1=result_x.iloc[22+10,6]
f1=result_x.iloc[22+15,6]

reversion1 = [a1,b1,c1,d1,e1,f1]
plt.plot(reversion1,'-ok',color='black')
plt.xlabel(['decision   ','filled   ',' 5min  ','10min','   15min','   20min'])






### 2: moving average algo:
###### Description and Assumption:
### take liquidity when 5minute sma crosses 10minute sma above
buy_remaining2=100
crossed_orders2=[]
decision_price=result_x.iloc[0,6]
### cost analysis
for i in range(9,len(result_x)):
    five_ma=result_x.iloc[i,7]
    ten_ma=result_x.iloc[i,8]
    offer_amount= result_x.iloc[i,1]
    if five_ma > ten_ma and buy_remaining2 > 0:
        if offer_amount<=buy_remaining2:
            buy_remaining2=buy_remaining2-offer_amount
            crossed_orders2.append(result_x.iloc[i,:])
        else:
            buy_remaining2=0
            crossed_orders2.append(result_x.iloc[i,:])
            crossed_orders2_df=pd.DataFrame(crossed_orders2)
            crossed_orders2_df.iloc[-1,1]=buy_amount-sum(crossed_orders2_df.iloc[:-1,1])
            
    elif buy_remaining2==0:
        print('orders all filled')
        break
#implementation shortfall calculation
is2=0
for i in range(len(crossed_orders2_df)):
    is2+=crossed_orders2_df.iloc[i,1]*(crossed_orders2_df.iloc[i,6]-decision_price)
is2_pct=is2/(100*decision_price)
is2_pct
#22.2 bp-> 23.2bp considering trading cost

#reversion analysis: price change curve
a2=decision_price
b2=crossed_orders1_df.iloc[-1,6]
c2=result_x.iloc[34,6]

d2=result_x.iloc[34+5,6]
e2=result_x.iloc[34+10,6]
f2=result_x.iloc[34+15,6]

reversion1 = [a2,b2,c2,d2,e2,f2]
plt.plot(reversion1,'-ok',color='black')
plt.xlabel(['decision   ','filled   ',' 5min  ','10min','   15min','   20min'])

#surprisingly, momentum strategy doesn't really catch the momentum, 
#probably because it is around open, things unpredictable, and IS is high, 
#failed compared to bench mark vwap algo





### 3: hidden liquidity seeking algo
###### Description and Assumption:
### use random forrest predict every order's probability of a hidden one,
### get average probabilities for every minute, when average probability is larger than certain amount we set 
### take all liquidity when there is a prediction: take 'vo' instead of 'vd'(the previous two algos)
### new_orderbook is generated by using the other .py file
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
# random forest(classification method)
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

new_orderbook= new_orderbook[new_orderbook['date']=='2006-04-03']

# time and hidden probability summing up assuming no correlations, summing up all hidden_prob
#to get hidden prob by minute
hidden_prob_by_minute=[]
for i in range(len(new_orderbook)):
    d=new_orderbook.iloc[i,2]
    t=new_orderbook.iloc[i,3]
    time=pd.Timestamp(year=d.year,month=d.month, day=d.day, hour=t.hour, minute=t.minute, second=0)
    templist=[time, new_orderbook.iloc[i,11]]
    hidden_prob_by_minute.append(templist)
hidden_prob_by_minute=pd.DataFrame(hidden_prob_by_minute, columns=['time','hidden_prob'])
hidden_prob_by_minute=hidden_prob_by_minute.set_index('time')
hidden_prob_by_minute.head(10)

agg_hidden=hidden_prob_by_minute.groupby(hidden_prob_by_minute.index)['hidden_prob'].mean().reset_index()
agg_hidden.head(10)

agg_hidden_list=agg_hidden['hidden_prob'].values
type(agg_hidden_list)

#histogram of average hidden probability in each minute
import matplotlib.pyplot as plt
plt.hist(agg_hidden_list)

#top 40% percentile
prob1=np.percentile(agg_hidden_list,60)
time_list1=agg_hidden[agg_hidden['hidden_prob']>prob1]['time'].reset_index()
time_list1=time_list1['time']
#0.18

#top 30% percentile
prob2=np.percentile(agg_hidden_list,70)
time_list2=agg_hidden[agg_hidden['hidden_prob']>prob2]['time'].reset_index()
time_list2=time_list2['time']
#0.21

#top 20% percentile
prob3=np.percentile(agg_hidden_list,80)
time_list3=agg_hidden[agg_hidden['hidden_prob']>prob3]['time'].reset_index()
time_list3=time_list3['time']
#0.25

#strategy:::

buy_remaining3=100
crossed_orders3=[]
decision_price=result_x.iloc[0,6]
time_list1_str=[]
for i in range(len(time_list1)):
    time_list1_str.append(time_list1[i].strftime('%Y-%m-%d %H:%M:%S'))
time_list2_str=[]
for i in range(len(time_list2)):
    time_list2_str.append(time_list2[i].strftime('%Y-%m-%d %H:%M:%S'))
time_list3_str=[]
for i in range(len(time_list3)):
    time_list3_str.append(time_list3[i].strftime('%Y-%m-%d %H:%M:%S'))
#test
#result_x.index[8].strftime('%Y-%m-%d %H:%M:%S') in time_list1_str

### cost analysis1
for i in range(len(result_x)):
    offer_amount= result_x.iloc[i,1]
    if result_x.index[i].strftime('%Y-%m-%d %H:%M:%S') in time_list1_str and buy_remaining3 > 0:#can be other time_list
        if offer_amount<=buy_remaining3:
            buy_remaining3=buy_remaining3-offer_amount
            crossed_orders3.append(result_x.iloc[i,:])
        else:
            buy_remaining3=0
            crossed_orders3.append(result_x.iloc[i,:])
    elif buy_remaining3==0:
        print('orders all filled')
        break

crossed_orders3_df=pd.DataFrame(crossed_orders3)
crossed_orders3_df.iloc[-1,1]=buy_amount-sum(crossed_orders3_df.iloc[:-1,1])

#implementation shortfall calculation
is3=0
for i in range(len(crossed_orders3_df)):
    is3+=crossed_orders3_df.iloc[i,1]*(crossed_orders3_df.iloc[i,6]-decision_price)
is3_pct=is3/(100*decision_price)
is3_pct
#18.5 bp->19.5bp

#reversion analysis: price change curve
a3=decision_price
b3=crossed_orders1_df.iloc[-1,6]
c3=result_x.iloc[27,6]

d3=result_x.iloc[27+5,6]
e3=result_x.iloc[27+10,6]
f3=result_x.iloc[27+15,6]

reversion1 = [a3,b3,c3,d3,e3,f3]
plt.plot(reversion1,'-ok',color='black')
plt.xlabel(['decision   ','filled   ',' 5min  ','10min','   15min','   20min'])
#interesting: liquidity algo finds momentum better than momentum strategy, open is probably exceptional
#guess: when buy_amout is larger, liquidity seeking works better
