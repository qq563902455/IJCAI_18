import pandas as pd
import numpy as np

trainData = pd.read_table(
    './rawdata/round2_train.txt', sep=' ', nrows=100000)


trainData['context_timestamp'] = pd.to_datetime(
    trainData['context_timestamp'], unit='s')

trainData = trainData.sort_values(by=['context_timestamp'])




def mergeIdRankInfo(dataset, collist):
    for col in collist:
        if type(col) == str:
            print('processing:\t', col)
            dataset = dataset.sort_values(by=['context_timestamp'])
            tempG = dataset.groupby(by=[col], as_index=False)
            dataset[col+'_rank'] = tempG.cumcount()
            tempTimeperiod = \
                pd.DataFrame(
                    (tempG['context_timestamp'].last()['context_timestamp']-\
                     tempG['context_timestamp'].first()['context_timestamp']
                    ).dt.seconds)
                
            tempTimeperiod[col] = tempG['context_timestamp'].last()[col]
            tempTimeperiod = tempTimeperiod.rename(
                    {'context_timestamp': col+'_timeperiod'}, axis=1)
            
            dataset = pd.merge(dataset, tempTimeperiod, on=[col])
            
    return dataset


mergeIdRankInfo(trainData, ['user_id'])[['user_id', 'context_timestamp', 'user_id_rank','user_id_timeperiod']].head(50)

mergeIdRankInfo(trainData, ['user_id']).columns

trainData[['user_id','context_timestamp']].head(20)

temp = trainData.groupby(by=['user_id'], as_index=False)
temp.cumcount().head(50)

temp['context_timestamp'].first()['user_id']
temp['context_timestamp'].last()['context_timestamp']
temp['context_timestamp']


(temp['context_timestamp'].last()['context_timestamp'] - \
    temp['context_timestamp'].first()['context_timestamp']).dt.seconds

 
 
temp['context_timestamp'].first()
temp.count().shape
temp['user_id'].last()
