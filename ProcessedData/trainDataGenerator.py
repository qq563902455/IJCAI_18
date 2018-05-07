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

data05 = allData[allData.day.isin(range(5))]
data6 = allData[allData.day == 6]
data7 = allData[allData.day == 7]

del allData
gc.collect()

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



def mergeIdRankInfo(dataset, collist):
    for col in collist:
        if type(col) == str:
            print('processing:\t', col)
            dataset = dataset.sort_values(by=['context_timestamp'])
            tempG = dataset.groupby(by=[col], as_index=False)
            dataset[col+'_rank'] = tempG.cumcount()
            dataset[col+'_rank'] /= (dataset[col+'_counts'] * dataset.shape[0])
            
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





info_list = []
# =============================================================================
# 生成头5天的特征
# =============================================================================
collist = ['item_brand_id', 'item_id', 'item_city_id', 'user_id',
           'user_occupation_id', 'user_gender_id', 'shop_id',
           'context_page_id', 'item_cat_id',
           ['user_id', 'item_id'], ['user_id', 'shop_id'],
           ['user_id', 'context_page_id'],
           ['user_id', 'item_cat_id'], ['user_gender_id', 'item_cat_id'],
           ['user_occupation_id', 'item_cat_id'], ['shop_id', 'item_cat_id'],
           ['user_id', 'hour'], ['item_id', 'hour'], ['shop_id', 'hour'],
           ]

for col in collist:
    if type(col) == str:
        info_list.append(getColInfo([col], data05,
                                    rate_col=True, nameAdd='05_'))
    elif type(col) == list:
        info_list.append(getColInfo(col, data05,
                                    rate_col=True, nameAdd='05_'))


del data05
gc.collect()
# =============================================================================
# 生成第6天的特征
# =============================================================================
collist = ['item_brand_id', 'item_id', 'item_city_id', 'user_id',
           'user_occupation_id', 'user_gender_id', 'shop_id',
           'context_page_id', 'item_cat_id',
           ['user_id', 'item_id'], ['user_id', 'shop_id'],
           ['user_id', 'context_page_id'],
           ['user_id', 'item_cat_id'], ['user_gender_id', 'item_cat_id'],
           ['user_occupation_id', 'item_cat_id'], ['shop_id', 'item_cat_id'],
           ['user_id', 'hour'], ['item_id', 'hour'], ['shop_id', 'hour'],
           ]

for col in collist:
    if type(col) == str:
        info_list.append(getColInfo([col], data6,
                                    rate_col=True, nameAdd='6_'))
    elif type(col) == list:
        info_list.append(getColInfo(col, data6,
                                    rate_col=True, nameAdd='6_'))
        
del data6
gc.collect()
# =============================================================================
# 生成第7天的特征
# =============================================================================
collist = ['item_brand_id', 'item_id', 'item_city_id', 'user_id',
           'user_occupation_id', 'user_gender_id', 'shop_id',
           'context_page_id', 'item_cat_id',
           ['user_id', 'item_id'], ['user_id', 'shop_id'],
           ['user_id', 'context_page_id'],
           ['user_id', 'item_cat_id'], ['user_gender_id', 'item_cat_id'],
           ['user_occupation_id', 'item_cat_id'], ['shop_id', 'item_cat_id'],
           ['user_id', 'hour'], ['item_id', 'hour'], ['shop_id', 'hour'],
           ]

for col in collist:
    if type(col) == str:
        info_list.append(getColInfo([col], data7))
    elif type(col) == list:
        info_list.append(getColInfo(col, data7))

colRanklist = ['user_id', 'item_id', 'item_brand_id', 'shop_id']


data7 = mergeInfo(data7, info_list)
del info_list
gc.collect()

data7 = mergeIdRankInfo(data7, colRanklist)
data7.to_csv('./ProcessedData/day.csv', index=False)
gc.collect()

#for day in [(6,0), (6,1), (7,0), (7,1)]:
#    print(day)
#    info_list = []
#    for col in collist:
#        if type(col) == str:
#            info_list.append(getColInfo([col], allData[
#                    np.array(allData.day == day[0])& 
#                    np.array(allData.isTestTime == day[1])]))
#        elif type(col) == list:
#            info_list.append(getColInfo(col, allData[
#                    np.array(allData.day == day[0])& 
#                    np.array(allData.isTestTime == day[1])]))
#
##    for col in collist:
##        if type(col) == str:
##            info_list.append(getColInfo([col], allData[allData.day == day-1],
##                                        rate_col=True, nameAdd='-1_'))
##        elif type(col) == list:
##            info_list.append(getColInfo(col, allData[allData.day == day-1],
##                                        rate_col=True, nameAdd='-1_'))
##
##    for col in collist:
##        if type(col) == str:
##            info_list.append(getColInfo([col], allData[allData.day == day-2],
##                                        rate_col=True, nameAdd='-2_'))
##        elif type(col) == list:
##            info_list.append(getColInfo(col, allData[allData.day == day-2],
##                                        rate_col=True, nameAdd='-2_'))
#
#
#
#    dataset = allData[
#        np.array(allData.day == day[0])& 
#        np.array(allData.isTestTime == day[1])]
#    dataset = mergeInfo(dataset, info_list)
#    
#    del info_list
#    gc.collect()
#    
#    dataset = mergeIdRankInfo(dataset, colRanklist)
#    dataset.to_csv(
#            './ProcessedData/day'+str(day[0])+'_'+str(day[1])+'.csv', index=False)
#    gc.collect()






# =============================================================================
# 这个模块的作用是把上面的那些day.csv转换成模型训练用的csv
# =============================================================================

data7 = pd.read_csv('./ProcessedData/day.csv')

catFeatureslist = ['item_id', 'item_brand_id', 'item_city_id',
                   'user_id', 'user_gender_id', 'user_occupation_id',
                   'context_id', 'shop_id',
                  ]

catDropList = ['item_id', 'item_brand_id',
               'user_id',
               'context_id', 'shop_id',
               'hour'
               ]

data7 = data7.drop(
    ['context_timestamp',
     'item_category_list',
     'item_property_list', 'predict_category_property'], axis=1)


data7 = data7.drop(catDropList, axis=1)

for col in data7.columns:
    if col in catFeatureslist:
        data7[col] = pd.Categorical(data7[col])

for col in data7:
    if col not in catFeatureslist + ['instance_id']:
        if col not in ['day', 'is_trade', 'isTestTime']:
            minval = data7[col].min()
            maxval = data7[col].max()
            if maxval != minval:
                data7[col] = data7[col].apply(
                    lambda x: (x-minval)/(maxval-minval))
            else:
                data7 = data7.drop([col], axis=1)





train = data7[data7.isTestTime==0].drop(['day', 'instance_id', 'isTestTime'], axis=1)
test = data7[data7.isTestTime == 1].drop(['is_trade', 'day',  'isTestTime'], axis=1)

#pretrain.to_csv('./ProcessedData/pretrain.csv', index=False)
#valid.to_csv('./ProcessedData/valid.csv', index=False)
train.to_csv('./ProcessedData/train.csv', index=False)
test.to_csv('./ProcessedData/test.csv', index=False)

gc.collect()
