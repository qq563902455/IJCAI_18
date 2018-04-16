import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
# %matplotlib inline

testData = pd.read_table(
    './rawdata/round1_ijcai_18_test_a_20180301.txt', sep=' ')
trainData = pd.read_table(
    './rawdata/round1_ijcai_18_train_20180301.txt', sep=' ')

allData = trainData.append(testData)
del testData
del trainData

allData['context_timestamp'] = pd.to_datetime(
    allData['context_timestamp'], unit='s')


allData.insert(loc=0, column='day', value=-1)
allData.insert(loc=0, column='hour', value=allData.context_timestamp.dt.hour)


def getStringVal(s, num):
    for i in range(num-1):
        if ';' in s:
            pos = s.index(';')
            s = s[pos+1:]
        else:
            return -1
    if ';' in s:
        pos = s.index(';')
        s = s[:pos]
    return s


allData.insert(loc=0, column='item_cat_id',
               value=allData['item_category_list'].apply(lambda x: getStringVal(x, 2)))
allData.insert(loc=0, column='item_cat_len',
               value=allData['item_category_list'].apply(lambda x: x.count(';')))


minTime = pd.to_datetime('2018-9-17 16:00:00')
for i in range(8):
    allData.loc[
        allData.context_timestamp >= minTime+pd.to_timedelta(str(i)+' days'),
        'day'] += 1


sampleDataList = []
for i in range(7):
    sampleDataList.append(allData[allData.day==i].sample(frac=0.33, random_state=2017))
sampleDataList.append(allData[allData.day==7])
sampleData = pd.concat(sampleDataList)
    

def getColInfo(col, dataset, rate_col=False, nameAdd=''):
    name = ''
    for i in col:
        name += i+'_'
    name += nameAdd

    table_info = dataset[col+['day']].groupby(by=col,
                                              as_index=False).count()
    if not rate_col:
        table_info = table_info.fillna(0)
        table_info = table_info.rename(columns={'day': name+'counts'})
    else:
        table_1_info = dataset[dataset['is_trade'] == 1][
            col+['day']].groupby(by=col, as_index=False).count()
        table_info = pd.merge(left=table_info, right=table_1_info, how='outer',
                              on=col)
        table_info = table_info.fillna(0)
        table_info.insert(loc=0, column=name+'rate', value=0)
        table_info.loc[table_info['day_y'] == 0, name+'rate'] = 0
        table_info.loc[table_info['day_y'] != 0, name+'rate'] = \
            table_info.loc[table_info['day_y'] != 0, 'day_y']\
            / table_info.loc[table_info['day_y'] != 0, 'day_x']

        table_info = table_info.drop(['day_y'], axis=1)
        # table_info = table_info.drop(['day_x'], axis=1)
        table_info = table_info.rename(columns={'day_x': name+'counts'})
        del table_1_info
    gc.collect()
    return table_info


def mergeInfo(rawData, datasetlist):
    for dataset in datasetlist:
        index_cols = []
        for col in rawData.columns:
            if col in dataset.columns:
                index_cols.append(col)

        rawData = pd.merge(left=rawData, right=dataset,
                           how='left', on=index_cols)
    rawData = rawData.fillna(0)
    return rawData


def mergeIdRankInfo(dataset, collist):
    for col in collist:
        print('processing:\t', col)
        idlist = list(dataset[col].unique())
        dataset[col+'_rank'] = 0
        for i in tqdm(range(len(idlist))):
            countNum = dataset[dataset[col]==idlist[i]][col].count()
            dataset.loc[dataset[col]==idlist[i], col+'_rank'] =\
                dataset[
                    dataset[col]==idlist[i]].context_timestamp.rank()/countNum
    return dataset

collist = ['item_brand_id', 'item_id', 'item_city_id', 'shop_id', 'user_id',
           'user_occupation_id', 'user_gender_id', 'shop_id',
           'context_page_id', 'context_id', 'item_cat_id',
           ['user_id', 'item_id'], ['user_id', 'shop_id'],
           ['user_id', 'item_cat_id'], ['user_gender_id', 'item_cat_id'],
           ['user_occupation_id', 'item_cat_id'], ['shop_id', 'item_cat_id'],
           ['user_occupation_id', 'item_cat_id']]

colRanklist = ['user_id', 'item_brand_id', 'item_id', 'shop_id']

for day in range(6,7):
    info_list = []
    for col in collist:
        if type(col) == str:
            info_list.append(getColInfo([col], sampleData[sampleData.day == day]))
        elif type(col) == list:
            info_list.append(getColInfo(col, sampleData[sampleData.day == day]))
            
    for col in collist:
        if type(col) == str:
            info_list.append(getColInfo([col], allData[allData.day == day-1],
                                        rate_col=True, nameAdd='-1_'))
        elif type(col) == list:
            info_list.append(getColInfo(col, allData[allData.day == day-1],
                                        rate_col=True, nameAdd='-1_'))
    for col in collist:
        if type(col) == str:
            info_list.append(getColInfo([col], allData[allData.day == day-2],
                                        rate_col=True, nameAdd='-2_'))
        elif type(col) == list:
            info_list.append(getColInfo(col, allData[allData.day == day-2],
                                        rate_col=True, nameAdd='-2_'))
            
    dataset = allData[allData.day == day]
    dataset = mergeInfo(dataset, info_list)
    
    if day == 6:
        temp_valid = dataset.copy().sample(frac=0.33, random_state=2017)
        temp_valid = mergeIdRankInfo(temp_valid, colRanklist)
        temp_valid.to_csv('./ProcessedData/temp_valid.csv', index=False)
    
    dataset = mergeIdRankInfo(dataset, colRanklist)
    dataset.to_csv(
            './ProcessedData/day'+str(day)+'.csv', index=False)
    gc.collect()