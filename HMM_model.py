__author__ = 'wangjia'
import numpy as np
import pandas as pd
import sys
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn import mixture
import matplotlib.pyplot as plt

from utility_function import ComputeEwave, HMMPrep, RFModel, RFWrapper

class HMM:
    def __init__(self,nState,TDFeaSet,featureSet,useAllFea,useDPGMM=True):
        '''
        Now only has one feature value for featureSet=['log_ewav_buy/sell']
        '''
		
        self.decay_factor = 0.001
        self.linger_factor = 5.0*self.decay_factor
        self.TDFeaSet = TDFeaSet
        self.featureSet = featureSet
        self.useDPGMM = useDPGMM
        self.useAllFea = useAllFea
        #self.df = data
        self.nState = nState
        self.tp = None
        self.pi = None
        self.TDmodel = []
        self.RatioModel = []

        self.state_map = [{'side':'B','IsSpoof':False},{'side':'S','IsSpoof':False},
                          {'side':'B','IsSpoof':True},{'side':'S','IsSpoof':True}]
						  
        self.fieldList = ['cancelled buy','cancelled sell', 'exec buy','exec sell','side','IsSpoof']

    def sanity_check(self,data):
        data_col = set(data.columns)
        dif = set(self.fieldList).difference(data_col)
        if len(dif) > 0:
            for xx in dif:
                print 'Data field {} is missing'.format(xx)
            sys.exit()

    def fitMixModel(self,data,showPara=True):
		''' fits the emission model for each state
		'''
		#if self.useDPGMM==True:
		#    dpgmm = mixture.DPGMM(n_components=5)
		#else:
		dpgmm = mixture.GMM(n_components=2)
		dpgmm.fit(data)

		if showPara:
			print '-------------------'
			print '------The mean------'
			print dpgmm.means_
			print '------The co-variance ---'
			print dpgmm.covars_
			print '------------------'
		return dpgmm
    def process_datafile(self,filename):
		data = pd.read_csv(filename)
		data = HMMPrep(data)
		data = self.DefStates(data)
		return data
		

    def DefStates(self,df):
		'''
		Use two columns from the training data: 'side' and 'IsSpoof'
		'''
		self.sanity_check(df)

		df['state'] = 0
		for ii in range(len(self.state_map)):
			df.loc[(df['side']==self.state_map[ii]['side'])&(df['IsSpoof']==self.state_map[ii]['IsSpoof']),'state'] = ii
		return df
		
    def train(self,df,show=False):
        self.pi = np.array(df.groupby('state').size()*1.0/len(df))
        
        df['next state'] = df['state'].shift(-1)    
        xx = pd.DataFrame()
        for dd in df['date'].unique():
            tmp = df.loc[df['date']==dd,:].copy()
            tmp = tmp.reset_index(drop=True)
            tmp = tmp.ix[0:len(tmp)-2]
            if len(tmp)<1:
                continue
            xx = xx.append(tmp)

        gp = xx.groupby(['state','next state','date']).size()
        aa = gp.sum(level=[0,1])
        bb = gp.sum(level=0)*1.
        self.tp = aa/bb

        print '---- Transition prob'
        print self.tp
        self.TDmodel = []
        self.RatioModel = []
        
        ### RF model for ratio #############
        
        ratio0 = df.loc[df['state']==0,self.featureSet]
        ratio2 = df.loc[df['state']==2,self.featureSet]
        rf_buy = RFModel(n_estimators=10,max_depth=2)
        rf_buy.fit(ratio0,0,ratio2,2)
        
        ratio1 = df.loc[df['state']==1,self.featureSet]
        ratio3 = df.loc[df['state']==3,self.featureSet]
        rf_sell = RFModel(n_estimators=10,max_depth=2)
        rf_sell.fit(ratio1,1,ratio3,3)
        
        self.RatioModel=[RFWrapper(rf_buy,0),RFWrapper(rf_sell,1),RFWrapper(rf_buy,2),RFWrapper(rf_sell,3)]   
        
        for state in range(self.nState):
            td = df.loc[df['state']==state,self.TDFeaSet]
            m1 = self.fitMixModel(td,showPara=False)
            self.TDmodel.append(m1)
			
        if show:
            self.plotDist2x(self.RatioModel[0],np.arange(-10,10,0.25),self.RatioModel[2],np.arange(-10,1e-5,0.25))
            self.plotDist2x(self.RatioModel[1],np.arange(-10,10,0.25),self.RatioModel[3],np.arange(1e-5,10,0.25))
			
    def plotDist2x(self,model1,x1,model2,x2):
		yy1 = model1.score(x1)
		yy2 = model2.score(x2)
		plt.plot(x1,yy1,'*',x2,yy2,'x')
		plt.show()
		
    def stateEstimator(self,obs):
		''' Estimate the most likely state sequence given the sequence of observations
		The data (obs) should be only one day data
		'''
		nState = self.nState
		TDmodel = self.TDmodel
		RatioModel = self.RatioModel
		tdlist = []
		rtlist = []

		for td in TDmodel:
			tdlist.append(td.score(obs[self.TDFeaSet]))
		for rt in RatioModel:    
			rtlist.append(rt.score(obs[self.featureSet]))

		tdprob = np.asmatrix(tdlist)
		rtprob = np.asmatrix(rtlist)
		if self.useAllFea == True:
			distrprob = tdprob + rtprob
		else:
			distrprob = rtprob

		logtp = np.log(tp)
		logpi = np.log(pi)

		backtrack = np.ones((nState,len(obs)))*(-1)
		pathscore = np.zeros((nState,len(obs)))

		isbuy = obs['side'].map(lambda x:int(x=='B'))
		issell = obs['side'].map(lambda x:int(x=='S'))
		validState = np.asmatrix([isbuy,issell,isbuy,issell]) # 0 means not valid
		dumbval = -1e30

		ttt = np.squeeze(np.asarray(distrprob[:,0])) + logpi
		pathscore[:,0] = ttt
		for ii in range(nState):
			if validState[ii,0]==0:
				pathscore[ii,0] = dumbval

		for ii in range(1,len(obs)):
			for jj in range(nState):
				tmp = logtp[:,jj] + pathscore[:,ii-1]+np.squeeze(np.asarray(distrprob[:,ii]))
				pathscore[jj,ii] = max(tmp)
				backtrack[jj,ii] = np.argmax(tmp)
			for kk in range(nState):
				if validState[kk,ii]==0:
					pathscore[kk,ii] = dumbval
					backtrack[kk,ii] = -1
		stateSeq = [-1]*len(obs)
		stateSeq[len(obs)-1] = np.argmax(pathscore[:,len(obs)-1])
		for nn in range(len(obs)-2,-1,-1):
			stateSeq[nn] = backtrack[stateSeq[nn+1],nn+1]
		return stateSeq
		
    def stateProb(self,obs):
		'''Give the estimate of the probablity of each state at each time instance
		'''
		tdlist = []
		rtlist = []

		nState = self.nState
		TDmodel = self.TDmodel
		RatioModel = self.RatioModel

		if False:
			print '----Debug Info------'
			self.testModelPrint(200)
			self.testModelPrint(-200)    


		for td in TDmodel:
			tdlist.append(td.score(np.array(obs[self.TDFeaSet])))
		for rt in RatioModel:    
			rtlist.append(list(rt.score(np.array(obs[self.featureSet])))) #low efficiency code

		#import pdb;pdb.set_trace()
		tdprob = np.asmatrix(tdlist)
		rtprob = np.asmatrix(rtlist)
		if self.useAllFea == True:
			distrprob = tdprob + rtprob
		else:            
			distrprob = rtprob        
		logtp = np.log(self.tp)
		logpi = np.log(self.pi)

		alpha = np.zeros((nState,len(obs)))
		beta = np.zeros((nState,len(obs)))

		isbuy = obs['side'].map(lambda x:int(x=='B'))
		issell = obs['side'].map(lambda x:int(x=='S'))
		validState = np.asmatrix([isbuy,issell,isbuy,issell]) # 0 means not valid
		dumb = -1e5 #used to fill for np.log(zero)

		alpha[:,0] = np.squeeze(np.asarray(distrprob[:,0])) + logpi
		for ii in range(1,len(obs)):
			for kk in range(nState):
				if validState[kk,ii]==0:
					alpha[kk,ii] = dumb
				else:
					tmp = alpha[:,ii-1] + logtp[:,kk]
					maxtmp = np.max(tmp)
					tmp = tmp - maxtmp
					alpha[kk,ii] = maxtmp + np.log(np.sum(np.exp(tmp))) + distrprob[kk,ii]

		for ii in range(len(obs)-2,-1,-1):
			for kk in range(nState):
				if validState[kk,ii] == 0:
					beta[kk,ii] = dumb
				else:
					tmp = np.asarray(logtp[kk])+beta[:,ii+1]+np.squeeze(np.asarray(distrprob[:,ii+1]))
					maxtmp = np.max(tmp)
					tmp = tmp - maxtmp
					beta[kk,ii] = maxtmp + np.log(np.sum(np.exp(tmp)))
			
		gamma = alpha+beta # not exactly the gamma
		maxgamma = np.max(gamma,0)
		gamma = gamma - np.kron(np.reshape(maxgamma,(1,len(obs))),np.ones((nState,1)))
		gamma = np.exp(gamma)
		sumgamma = np.kron(np.sum(gamma,0),np.ones((nState,1)))
		gamma = gamma/sumgamma   
		return gamma
		
    def predict(self,df):
		''' needs more work,better return a dataframe
		'''
		#import pdb;pdb.set_trace()
		res = pd.DataFrame()
		for xx in df['date'].unique():
			data = df.loc[df['date']==xx,:].copy()
			prob = self.stateProb(data)
			pred = np.argmax(prob,0)
			pred_prob=np.max(prob,0)
			data['pred'] = pred
			data['pred_prob'] = pred_prob
			data['predSpoofing'] = data['pred'].map(lambda x:x>1)
			res = res.append(data)
		return res
    
    def test(self,df):
		#import pdb;pdb.set_trace()
		all_truth = []
		all_score = []

		all_pred = []
		all_state = []

		res = pd.DataFrame()

		for xx in df['date'].unique():
			data = df.loc[df['date']==xx,:].copy()
			#import pdb;pdb.set_trace()
			prob = self.stateProb(data)
			pred = np.argmax(prob,0)
			pred_prob=np.max(prob,0)
			data['pred']=pred
			data['pred_prob']=pred_prob
			
			tmp = (np.array(pred) == np.array(data['state']))
			r = sum(tmp)*1.0/len(tmp)
			truth = map(lambda x:int(x>1),np.array(data['state']))
			score = [y if x>1 else 1-y for x,y in zip(pred,pred_prob)]
			#import pdb;pdb.set_trace()
			auc = metrics.roc_auc_score(truth,score)
			all_truth = all_truth + truth
			all_score = all_score + score
			all_pred = all_pred + list(pred)
			all_state = all_state + list(data['state'])
			res = res.append(data)
		#import pdb;pdb.set_trace()
		auc = metrics.roc_auc_score(all_truth,all_score)
		tmp = (np.array(all_pred) == np.array(all_state))
		rate = sum(tmp)*1./len(tmp)
		return auc,rate,res