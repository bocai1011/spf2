__author__ = 'wangjia'

from feature import Feature
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class FeatureCombiner:
    def __init__(self):
        #self.fea_data = None
        self.fea_list = []

    def add_feature(self,fea):
        self.fea_list.append(fea)

    def calculate_all_feature(self,data):
        fea_list = []
        for ff in self.fea_list:
            fea_list.append(ff.fea_computation(data))
        fea_data = pd.concat(fea_list)
        return fea_data

    def fit(self,data):
        pass

    def pred_prob(self,data):
        pass

    def pred_log_prob(self,data):
        pass


class RandomForestCombiner(FeatureCombiner):
    def __init__(self, n_estimators, max_depth):
        FeatureCombiner.__init__(self)
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.label_map = {}
        self.rev_map = {}

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
        plt.plot(tt, proba[:,0],color='b')
        plt.plot(tt, proba[:,1],color='r')
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