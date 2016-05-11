import pandas as pd
import numpy as np
import sys

def process_data(df,sym):
    df = df[df['symbol']== sym]
    df = df[['acct','date','ordid','time','symbol','mtype','side','limitprice','ordqty','trdprice','trdqty']]
    df = df[df['mtype'].isin(['CancelRequest','Filled','PartialFill'])]
    df['mtype'] = df['mtype'].replace(['CancelRequest','Filled','PartialFill'],['CANCEL','EXECUTION','EXECUTION'])
    df['time'] = pd.to_datetime(df['time'].str[2:])
    df['time_diff'] = df['time'].diff()
    df['time']=df['time'].map(lambda x:x.time())
    T = df['time_diff'].median()
    df['time_diff'].iat[0] = T
    df['time_diff'] = df['time_diff'].map(lambda x:x.microseconds)
    df['cancelled buy'] = df['ordqty']
    df['cancelled sell'] = df['ordqty']
    df['exec buy'] = df['trdqty']
    df['exec sell'] = df['trdqty']
    df.loc[(df['mtype']!='CANCEL')|(df['side']!='BY'),'cancelled buy'] = 0.0
    df.loc[(df['mtype']!='CANCEL')|(df['side']!='SL'),'cancelled sell'] = 0.0
    df.loc[(df['mtype']!='EXECUTION')|(df['side']!='BY'),'exec buy'] = 0.0
    df.loc[(df['mtype']!='EXECUTION')|(df['side']!='SL'),'exec sell'] = 0.0
    df.sort_values('time')
    df = df[['ordid','time','symbol','mtype','time_diff','cancelled buy','cancelled sell','exec buy','exec sell']]
    df.index = range(len(df))
    return df
	
def calc_feature(data):
    decay_factor = 2.0
    linger_factor = 100.0
    if len(data)<2:
        raise ValueError('data too short')
    medianT = data['time_diff'].median()
    T=medianT*decay_factor
    linger = medianT*linger_factor
    epsilon = sys.float_info.epsilon
    
    data['ewav_canc_buy'] = epsilon
    data['ewav_canc_sell'] = epsilon

    for ii in range(1,len(data)):
        coef = np.exp(-data.ix[ii]['time_diff']/T) if data.ix[ii]['time_diff'] <= linger else 0
        #import pdb;pdb.set_trace()
        data.loc[ii,'ewav_canc_buy'] = data.loc[ii, 'cancelled buy']+data.loc[ii-1, 'ewav_canc_buy']*coef
        data.loc[ii,'ewav_canc_sell'] = data.loc[ii, 'cancelled sell']+data.loc[ii-1, 'ewav_canc_sell']*coef
    
    ff = lambda x: x if x > epsilon else epsilon
    data['ewav_canc_buy'] = data['ewav_canc_buy'].map(ff)
    data['ewav_canc_sell'] = data['ewav_canc_sell'].map(ff)
    
    data['ewav_buy/sell'] = data['ewav_canc_buy']/data['ewav_canc_sell']
    data['log_ewav_buy/sell'] = np.log(data['ewav_buy/sell'])
    data['ewav_sell/buy'] = data['ewav_canc_sell']/data['ewav_canc_buy']
    data['log_ewav_sell/buy'] = np.log(data['ewav_sell/buy'])
    data = data[['ordid','time','symbol','mtype','ewav_buy/sell','log_ewav_buy/sell','ewav_sell/buy','log_ewav_sell/buy']]
    data.rename(columns={'mtype':'order_type'},inplace=True)
    return data[data['order_type']=='EXECUTION']
	

def sort_syms(df):
    gp = df.groupby('symbol').size()
    syms = gp[gp>10000].index.values
    ret = []
    for sym in syms:
        tmp = df[df['symbol']==sym]
        unique_mtypes = tmp['mtype'].unique()
        if ('Filled' in unique_mtypes ) or ('PartialFill' in unique_mtypes):
            ret.append(sym)
    return ret
		
df =  pd.read_csv('spoof20150610.csv')
syms = sort_syms(df)
data = process_data(df,syms[0])
fea = calc_feature(data)

