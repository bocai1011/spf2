"""
    Function:
    1. perform basic sanity check
    2. provide basic utilities to manipulate the data
    3. provide a basic version to generate the data that human can label
    Assuming data is in the format of sina0708.csv
"""

__author__ = 'Jianqi Wang'
import pandas as pd
import numpy as np
import sys


def toMS(x):
    return ((x.hour*60+x.minute)*60+x.second)*1000000+x.microsecond
def timeDelta2MS(x):
    return (((x.hours*60+x.minutes)*60+x.seconds)*1000+x.milliseconds)*1000+x.microseconds


class InputProcessing:

    def __init__(self):
        self.fieldList = set(['date', 'q_exec', 'q_new', 'q_canc', 'execution_time', 'cancel_entry_time',
                          'order_entry_time', 'prc_exec', 'order_type', 'id', 'orderid', 'symbol', 'q_new',
                          'price', 'time', 'side'])

    def sanity_check(self, data):
        data_col = set(data.columns)
        dif = self.fieldList.difference(data_col)
        if len(dif) > 0:
            for xx in dif:
                print 'Data field {} is missing'.format(xx)
            #import pdb;pdb.set_trace()
            sys.exit()

    def data_for_label(self,infile,outfile):
        allorder = self.process_datafile(infile)
        allorder.to_csv(outfile,index=False)

    def process_datafile(self, filename):
        data = pd.read_csv(filename)
        return self.process_data(data)

    def process_data(self,data):
        self.sanity_check(data)
        allorder = self.data_cleaning(data)
        allorder = self.re_arrange(allorder)
        return allorder

    def data_cleaning(self,data):
        # clean the data
        data['q_exec'].fillna(0,inplace=True)
        data['execution_time'] = data['execution_time'].map(lambda x: pd.to_datetime(x))
        data['cancel_entry_time'] = data['cancel_entry_time'].map(lambda x:pd.to_datetime(x))
        data['order_entry_time'] = data['order_entry_time'].map(lambda x:pd.to_datetime(x))
        data['prc*qty'] = data['q_exec']*data['prc_exec']

        neworder = data.loc[data['order_type'] == 'NEW ORDER', :]
        exeorder = data.loc[data['order_type'] == 'EXECUTION', :]
        canorder = data.loc[data['order_type'] == 'CANCEL', :].copy()

        ############## Exclude those partial filled orders from cancel list
        #partialfill = set(canorder['orderid']).intersection(set(exeorder['orderid']))
        #canorder = canorder.loc[canorder['orderid'].isin(partialfill)==False,:]
        #####################################################################

        allorder = neworder[['id','orderid','symbol','q_new','price','order_entry_time','date','time','side']].set_index('orderid')
        gp = exeorder.groupby('orderid')
        tmp = gp.agg({'q_exec': np.sum,'prc*qty': np.sum})
        tmp['avg exe_prc'] = tmp['prc*qty']/tmp['q_exec']
        del tmp['prc*qty']
        allorder = allorder.join(tmp)
    #allorder = allorder.join(gp['execute_time'].agg({'first_exe_time':np.min,'last_exe_time':np.max}))
        allorder = allorder.join(gp['execution_time'].agg({'first_execution_time':np.min,'last_execution_time':np.max}))
        allorder['execution_time_first_ms'] = allorder['first_execution_time'].map(toMS)
        allorder['execution_time_last_ms'] = allorder['last_execution_time'].map(toMS)
        allorder['order_entry_time_ms'] = allorder['order_entry_time'].map(toMS)
    #gp = canorder.groupby('orderid')
        allorder = allorder.join(canorder.set_index('orderid')[['cancel_entry_time','canc_time']])
        allorder['q_exec'].fillna(0,inplace=True)
        allorder['q_cancel'] = allorder['q_new'] - allorder['q_exec']
        allorder = allorder.sort('order_entry_time')
        return allorder

    def re_arrange(self,allorder):
        # resort all the order according the the order entry time (canceled order) and exe time(filled order)
        #    Calculate the time difference
        #
        fillorder = allorder.loc[allorder['q_exec']>0,['date','price','side','last_execution_time','execution_time_last_ms','q_exec']]
        fillorder['exec buy'] = fillorder['q_exec']
        fillorder['exec sell'] = fillorder['q_exec']
        fillorder.loc[fillorder['side']=='B','exec sell'] = 0
        fillorder.loc[fillorder['side']!='B','exec buy'] = 0
        fillorder = fillorder.rename(columns={'execution_time_last_ms':'microsecond','last_execution_time':'time'})

    #canorder = allorder.loc[allorder['q_cancel']>0,['date','price','side','order_entry_time','order_entry_time_ms','q_cancel']]
        canorder = allorder.loc[allorder['q_cancel']==allorder['q_new'],['date','price','side','order_entry_time','order_entry_time_ms','q_cancel']]
    #partially filled order discarded
    #import pdb;pdb.set_trace()
        canorder['cancelled buy'] = canorder['q_cancel']
        canorder['cancelled sell'] = canorder['q_cancel']
        canorder.loc[canorder['side']=='B','cancelled sell'] = 0.0
        canorder.loc[canorder['side']!='B','cancelled buy'] = 0.0
        canorder = canorder.rename(columns={'order_entry_time_ms':'microsecond','order_entry_time':'time'})

        fillorder['cancelled buy'] = 0
        fillorder['cancelled sell'] = 0
        canorder['exec buy'] = 0
        canorder['exec sell'] = 0
        del canorder['q_cancel']
        del fillorder['q_exec']
        data = fillorder.append(canorder)
        data = data.sort(['date','microsecond'])
        data = data.reset_index()
        #import pdb;pdb.set_trace()
        for dd in data['date'].unique():
            data.loc[data['date']==dd,'inventory'] = data.loc[data['date']==dd,'exec buy']-data.loc[data['date']==dd,'exec sell']
            data.loc[data['date']==dd,'inventory'] = data.loc[data['date']==dd,'inventory'].cumsum()
            data.loc[data['date']==dd,'time diff']= data.loc[data['date']==dd,'microsecond'].diff()*1.
        data['time diff'] = data['time diff'].fillna(24*3600*1000000)
        return data