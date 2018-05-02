import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import gc
#from tqdm import tqdm
import random


random.seed(10)
flag = False


def rowRandomSelect(x):
    global flag
    if not flag:
        print('a')
        flag = True
        return False
    else:
        return random.random() > 0.3


testData = pd.read_table(
    './rawdata/round2_ijcai_18_test_a_20180425.txt', sep=' ')


trainData = pd.read_table(
    './rawdata/round2_train.txt', sep=' ', skiprows=rowRandomSelect)


allData = trainData.append(testData)
del testData
del trainData
gc.collect()

allData['context_timestamp'] = pd.to_datetime(
    allData['context_timestamp'], unit='s')

allData.context_timestamp += pd.to_timedelta('8:00:00')

allData.insert(loc=0, column='day', value=allData.context_timestamp.dt.day)
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
               value=allData['item_category_list'].apply(
                lambda x: getStringVal(x, 2)))
allData.insert(loc=0, column='item_cat_len',
               value=allData['item_category_list'].apply(
                lambda x: x.count(';')))

allData.insert(loc=0, column='isTestTime', value=0)
allData.loc[allData.hour >= 12, 'isTestTime'] = 1


def getColInfo(col, dataset, rate_col=False, nameAdd=''):
    name = ''
    for i in col:
        name += i+'_'
    name += nameAdd

    table_info = dataset[col+['day']].groupby(
            by=col, as_index=False).count()

    if not rate_col:
        table_info = table_info.fillna(0)
        table_info = table_info.rename(columns={'day': name+'counts'})
        table_info[name+'counts'] /= dataset.shape[0]
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


def getColMeanInfo(col, dataset, meanCol, nameAdd=''):
    name = ''
    for i in col:
        name += i+'_'
    name += meanCol

    name += nameAdd
    table_info = dataset[col+[meanCol]].groupby(by=col,
                                              as_index=False).mean()

    table_info = table_info.fillna(0)
    table_info = table_info.rename(columns={meanCol: name+'_mean'})

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





index = 0
lastId = -100
def getRankInfo(row, col):
    global index
    global lastId
    if row[col] != lastId:
        index=0
    index += 1
    lastId = row[col]
    return index

lastId = -100
lastTime = -1
def getDiffTimeInfo(row, col):
    global lastTime
    global lastId
    if row[col] != lastId:
        lastTime = row['context_timestamp']
        lastId = row[col]
        return 0
    re = row['context_timestamp'] - lastTime
    lastId = row[col]
    lastTime = row['context_timestamp']
    return re.seconds

def mergeIdRankInfo(dataset, collist):
    global index
    global lastId
    global lastTime
    for col in collist:
        if type(col) == str:
            print('processing:\t', col)
            dataset = dataset.sort_values(by=[col, 'context_timestamp'])
            index = 0
            lastId = -100
            lastTime = -1
            dataset[col+'_rank'] = dataset.apply(
                    lambda x: getRankInfo(x, col), axis=1)
            
            dataset[col+'_rank'] = dataset[col+'_rank']/dataset[col+'_counts']
            
            
            index = 0
            lastId = -100
            lastTime = -1
            dataset[col+'_difftime'] = dataset.apply(
                    lambda x: getDiffTimeInfo(x, col), axis=1)
            
            gc.collect()
            
    return dataset


#lastTime = -1
#def getLastTimeDiff(x):
#    global lastTime
#    if type(lastTime) == int:
#        lastTime = x
#        return -1
#    temp = lastTime
#    lastTime = x
#    return (x-temp).seconds
#
#
#def mergeIdTimeDiffInfo(dataset, collist):
#    global lastTime
#    dataset = dataset = dataset.sort_values(by='context_timestamp')
#    for col in collist:
#        print('processing:\t', col)
#        idlist = list(dataset[col].unique())
#        dataset[col+'_timeDiff'] = 0
#        for i in tqdm(range(len(idlist))):
#            lastTime = -1
#            dataset.loc[dataset[col]==idlist[i], col+'_timeDiff'] =\
#                    dataset[
#                        dataset[col]==idlist[i]].context_timestamp.apply(getLastTimeDiff)
#    return dataset

collist = ['item_brand_id', 'item_id', 'item_city_id', 'user_id',
           'user_occupation_id', 'user_gender_id', 'shop_id',
           'context_page_id', 'context_id', 'item_cat_id',
           ['user_id', 'item_id'], ['user_id', 'shop_id'],
           ['user_id', 'item_cat_id'], ['user_gender_id', 'item_cat_id'],
           ['user_occupation_id', 'item_cat_id'], ['shop_id', 'item_cat_id'],
           ]

colRanklist = ['user_id', 'item_brand_id', 'item_id', 'shop_id']





for day in range(6, 8):
    print(day)
    info_list = []
    for col in collist:
        if type(col) == str:
            info_list.append(getColInfo([col], allData[allData.day == day]))
        elif type(col) == list:
            info_list.append(getColInfo(col, allData[allData.day == day]))

#    for col in collist:
#        if type(col) == str:
#            info_list.append(getColInfo([col], allData[allData.day == day-1],
#                                        rate_col=True, nameAdd='-1_'))
#        elif type(col) == list:
#            info_list.append(getColInfo(col, allData[allData.day == day-1],
#                                        rate_col=True, nameAdd='-1_'))
#
#    for col in collist:
#        if type(col) == str:
#            info_list.append(getColInfo([col], allData[allData.day == day-2],
#                                        rate_col=True, nameAdd='-2_'))
#        elif type(col) == list:
#            info_list.append(getColInfo(col, allData[allData.day == day-2],
#                                        rate_col=True, nameAdd='-2_'))



    dataset = allData[allData.day == day]
    dataset = mergeInfo(dataset, info_list)
    
    del info_list
    gc.collect()
    
    dataset = mergeIdRankInfo(dataset, colRanklist)
    dataset.to_csv(
            './ProcessedData/day'+str(day)+'.csv', index=False)
    gc.collect()






# =============================================================================
# 这个模块的作用是把上面的那些day.csv转换成模型训练用的csv
# =============================================================================
datalist = []
for i in range(6, 8):
    datalist.append(pd.read_csv('./ProcessedData/day'+str(i)+'.csv'))

selectedOutId = pd.read_table(
    './rawdata/round2_ijcai_18_test_a_20180425.txt', sep=' ').instance_id

outId = datalist[0].instance_id
allData = pd.concat(datalist)


catFeatureslist = ['item_id', 'item_brand_id', 'item_city_id',
                   'user_id', 'user_gender_id', 'user_occupation_id',
                   'context_id', 'shop_id',
                  ]

catDropList = ['item_id', 'item_brand_id', 'item_city_id',
               'user_id', 'user_gender_id', 'user_occupation_id',
               'context_id', 'shop_id',
               'hour'
               ]

allData = allData.drop(
    ['context_timestamp',
     'item_category_list', 'instance_id',
     'item_property_list', 'predict_category_property'], axis=1)


allData = allData.drop(catDropList, axis=1)

for col in allData.columns:
    if col in catFeatureslist:
        allData[col] = pd.Categorical(allData[col])

for col in allData:
    if col not in catFeatureslist:
        if col not in ['day', 'is_trade']:
            minval = allData[col].min()
            maxval = allData[col].max()
            if maxval != minval:
                allData[col] = allData[col].apply(
                    lambda x: (x-minval)/(maxval-minval))
            else:
                allData = allData.drop([col], axis=1)






train = allData[allData.day==6].drop(['isTestTime', 'day'], axis=1)
valid = allData[np.array(allData.isTestTime == 0)&
                np.array(allData.day==7)].drop(['isTestTime', 'day'], axis=1)
test = allData[np.array(allData.isTestTime == 1)&
               np.array(allData.day==7)].drop(['is_trade', 'isTestTime', 'day'], axis=1)

#pretrain.to_csv('./ProcessedData/pretrain.csv', index=False)
valid.to_csv('./ProcessedData/valid.csv', index=False)
train.to_csv('./ProcessedData/train.csv', index=False)
test.to_csv('./ProcessedData/test.csv', index=False)

gc.collect()
