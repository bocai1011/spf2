__author__ = 'wangjia'

import numpy as np
import math
import sys

class Feature:
    def __init__(self,decay_factor,linger_factor=None):
        self.fieldList = ['time diff', 'cancelled buy','cancelled sell', 'exec buy']
        self.fea_name_list = ['ewav_buy/sell', 'log_ewav_buy/sell', 'ewav_sell/buy']
        self.decay_factor = decay_factor
        if linger_factor is None:
            self.linger_factor = 5*self.decay_factor
        else:
            self.linger_factor = linger_factor
        self.medianT = 0
        self.fea_data = None

    def sanity_check(self,data):
        data_col = set(data.columns)
        dif = set(self.fieldList).difference(data_col)
        if len(dif) > 0:
            for xx in dif:
                print 'Data field {} is missing'.format(xx)
            sys.exit()

    def fea_computation(self, in_data):
        # exponentially weighted volume
        if len(in_data)<2:
            raise ValueError('data too short')
        self.sanity_check(in_data)

        #data = in_data[self.fieldList].copy()
        data = in_data
        data['time diff'] = data['time diff'].fillna(24*3600*1000000)
        self.medianT = np.median(data['time diff'])
        T = self.medianT*self.decay_factor
        linger = self.medianT*self.linger_factor
        epsilon = sys.float_info.epsilon

        data['ewav_canc_buy'] = epsilon
        data['ewav_canc_sell'] = epsilon

        for ii in range(1,len(data)):
            coef = math.exp(-data.ix[ii]['time diff']/T) if data.ix[ii]['time diff'] <= linger else 0
            data.loc[ii,'ewav_canc_buy'] = data.loc[ii, 'cancelled buy']+data.loc[ii-1, 'ewav_canc_buy']*coef
            data.loc[ii,'ewav_canc_sell'] = data.loc[ii, 'cancelled sell']+data.loc[ii-1, 'ewav_canc_sell']*coef

        ff = lambda x: x if x > epsilon else epsilon
        data['ewav_canc_buy'] = data['ewav_canc_buy'].map(ff)
        data['ewav_canc_sell'] = data['ewav_canc_sell'].map(ff)

        data['ewav_buy/sell'] = data['ewav_canc_buy']/data['ewav_canc_sell']
        data['log_ewav_buy/sell'] = data['ewav_buy/sell'].map(math.log)
        data['ewav_sell/buy'] = data['ewav_canc_sell']/data['ewav_canc_buy']

        private_columns = ['ewav_buy/sell','log_ewav_buy/sell','ewav_sell/buy'] #The feature data inside the class has more columns than it outputs
        self.fea_data = data[private_columns].copy()
        return data

