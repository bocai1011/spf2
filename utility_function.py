import numpy as np
import pandas as pd
import sys
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn import mixture
import matplotlib.pyplot as plt

def ComputeEwave(data):
	if len(data)<2:
		raise ValueError('data too short')
		
	decay_factor = 0.001
	linger_factor = 0.005

	data['time diff'] = data['time diff'].fillna(24*3600*1000000)
	medianT = np.median(data['time diff'])
	T = medianT*decay_factor
	linger = medianT*linger_factor
	epsilon = sys.float_info.epsilon

	data['ewav_back canc buy'] = epsilon
	data['ewav_back canc sell'] = epsilon
	data['ewav_back exec buy'] = epsilon
	data['ewav_back exec sell'] = epsilon

	for ii in range(1,len(data)):
		coef = math.exp(-data.ix[ii]['time diff']/T) if data.ix[ii]['time diff']<=linger else 0
		data.loc[ii,'ewav_back canc buy'] = data.loc[ii,'cancelled buy']+data.loc[ii-1,'ewav_back canc buy']*coef
		data.loc[ii,'ewav_back canc sell'] = data.loc[ii,'cancelled sell']+data.loc[ii-1,'ewav_back canc sell']*coef
		data.loc[ii,'ewav_back exec buy'] = data.loc[ii,'exec buy']+data.loc[ii-1,'ewav_back exec buy']*coef
		data.loc[ii,'ewav_back exec sell'] = data.loc[ii,'exec sell']+data.loc[ii-1,'ewav_back exec sell']*coef
		
	ff = lambda x:x if x>epsilon else epsilon
	data['ewav_back canc buy'] = data['ewav_back canc buy'].map(ff)
	data['ewav_back canc sell'] = data['ewav_back canc sell'].map(ff)
	data['ewav_back exec buy'] = data['ewav_back exec buy'].map(ff)
	data['ewav_back exec sell'] = data['ewav_back exec sell'].map(ff)

	data['ewav_back buy/sell'] = data['ewav_back canc buy']/data['ewav_back canc sell'] 
	data['log ewav_back buy/sell'] = data['ewav_back buy/sell'].map(math.log)
	data['ewav_back sell/buy'] = data['ewav_back canc sell']/data['ewav_back canc buy']
			
	data['ewav_back buy exec+canc'] = data['ewav_back exec buy'] + data['ewav_back canc buy']
	data['ewav_back buy exec/total']=  data['ewav_back exec buy']/data['ewav_back buy exec+canc']       

	data['ewav_back sell exec+canc'] = data['ewav_back exec sell'] + data['ewav_back canc sell']
	data['ewav_back sell exec/total'] = data['ewav_back exec sell']/data['ewav_back sell exec+canc']

	return data
	
def HMMPrep(df):
	#import pdb;pdb.set_trace()
	df = df.copy()
	col = ['orderid','cancelled buy','exec sell','cancelled sell','exec buy','microsecond','price','side','time','date','inventory','time diff','ewav_back canc buy','ewav_back canc sell','ewav_back exec buy','ewav_back exec sell','ewav_back buy/sell','ewav_back sell/buy']
	if 'IsSpoof' in df.columns:
		col +=['IsSpoof']
	df = df[col]
	del df['ewav_back exec buy']
	del df['ewav_back exec sell']
	# clean the data for ewav_back canc buy/sell and sell/buy
	# buy/sell will be just inverse of sell/buy, so we use one column buy/sell
	df.loc[(df['ewav_back canc buy']<1e-5)&(df['ewav_back canc sell']<1e-5),'ewav_back buy/sell']=1
	medianbs = df.loc[(df['ewav_back buy/sell']>0)&(df['ewav_back buy/sell']<np.inf),'ewav_back buy/sell'].median()
	maxbs = df.loc[(df['ewav_back buy/sell']>0)&(df['ewav_back buy/sell']<np.inf),'ewav_back buy/sell'].max()
	df.loc[df['ewav_back buy/sell']==np.inf,'ewav_back buy/sell'] = maxbs
	df.loc[df['ewav_back buy/sell']==0,'ewav_back buy/sell'] = 1/maxbs
	df.loc[:,'ewav_back buy/sell'] = df.loc[:,'ewav_back buy/sell'].map(np.log)

	## Get the time difference, seems not contributing for now
	df['TimeDiff_back'] = np.nan
	df['TimeDiff_frwd'] = np.nan
	df['TimeDiff_min'] = np.nan
	#import pdb;pdb.set_trace()

	df = df.loc[(df['exec sell']>0)|(df['exec buy']>0),:].copy()
	if len(df)==0:
		return df
	buy = df.loc[df['side']=='B',:].copy()
	if len(buy)>0:
		for dd in buy['date'].unique():
		#import pdb;pdb.set_trace()
			tmp = buy.loc[buy['date']==dd,:]
			buy.loc[buy['date']==dd,'TimeDiff_back'] = buy.loc[buy['date']==dd,'microsecond'].diff(1).map(lambda x:np.abs(x))
			buy.loc[buy['date']==dd,'TimeDiff_frwd'] = buy.loc[buy['date']==dd,'microsecond'].diff(-1).map(lambda x:np.abs(x))
		#import pdb;pdb.set_trace()    
		buy['TimeDiff_frwd'].fillna(buy['TimeDiff_frwd'].max(),inplace=True)    
		buy['TimeDiff_back'].fillna(buy['TimeDiff_back'].max(),inplace=True)
		buy['TimeDiff_min'] = buy.apply(lambda x:min(x['TimeDiff_back'],x['TimeDiff_frwd']),axis=1)

	sell = df.loc[df['side']=='S',:].copy()
	if len(sell)>0:
		for dd in sell['date'].unique():
			tmp = sell.loc[sell['date']==dd,:]
			sell.loc[sell['date']==dd,'TimeDiff_back'] = sell.loc[sell['date']==dd,'microsecond'].diff(1).map(lambda x:np.abs(x))
			sell.loc[sell['date']==dd,'TimeDiff_frwd'] = sell.loc[sell['date']==dd,'microsecond'].diff(-1).map(lambda x:np.abs(x))

		sell['TimeDiff_frwd'].fillna(sell['TimeDiff_frwd'].max(),inplace=True)
		sell['TimeDiff_back'].fillna(sell['TimeDiff_back'].max(),inplace=True)
		sell['TimeDiff_min'] = sell.apply(lambda x:min(x['TimeDiff_back'],x['TimeDiff_frwd']),axis=1)

	newdf = buy.append(sell)
	newdf['date'] = newdf['date'].map(lambda x:pd.to_datetime(x))
	#newdf = newdf.sort(['date','microsecond'])
	df = newdf.sort()

	return df
	
class RFModel:
	def __init__(self,n_estimators,max_depth):
		self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
		self.label_map = {}
		self.rev_map = {}
		
		#self.label_set = label_set
		
	def fit(self,x1,label1,x2,label2):
		''' we assume x1,x2 are numpy arrays (1-d)
		'''
	   
		label = np.array([0]*len(x1)+[1]*len(x2))
		self.label_map = {0:label1,1:label2}
		self.rev_map ={label1:0,label2:1}
		data = np.concatenate((x1,x2)).reshape((len(x1)+len(x2),1))
		self.rf.fit(data,label)
		
		if True:
			self.showResult(x1,label1,x2,label2)

	def showResult(self,x1,label1,x2,label2):
		#import pdb;pdb.set_trace()
		plt.hist(np.array(x1),bins=100,alpha=0.5,normed=True)
		plt.hist(np.array(x2),bins=100,alpha=0.5,normed=True)
		tt = np.arange(-50,50,0.05)
		tt = tt.reshape((len(tt),1))
		proba = self.rf.predict_proba(tt)
		plt.plot(tt,proba[:,0],color='b')
		plt.plot(tt,proba[:,1],color='r')
		plt.show()
		
	def score(self,x,label):
		''' give score in log prob for the class denoted by label
		'''
		proba = self.rf.predict_proba(np.array(x).reshape((len(x),1)))
		return np.log(proba[:,self.rev_map[label]])

	def prob(self,x,label):
		''' give score in log prob for the class denoted by label
		'''
		proba = self.rf.predict_proba(np.array(x).reshape((len(x),1)))
		return proba[:,self.rev_map[label]]
     
class RFWrapper():
	def __init__(self,rf,label):
		self.rf = rf
		self.label= label
	def score(self,x):
		return self.rf.score(x,self.label)
	def prob(self,x):
		return self.rf.prob(x,self.label)