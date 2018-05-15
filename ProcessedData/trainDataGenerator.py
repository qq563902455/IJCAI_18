import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import gc
# from tqdm import tqdm
import random

random.seed(10)
flag = False

# 用于随机抽取数据
# def rowRandomSelect(x):
#     global flag
#     if not flag:
#         print('a')
#         flag = True
#         return False
#     else:
#         return random.random() > 0.3

# 读取test数据
testData = pd.read_table(
    './rawdata/round2_ijcai_18_test_a_20180425.txt', sep=' ')

testData = testData.append(pd.read_table(
    './rawdata/round2_ijcai_18_test_b_20180510.txt', sep=' '))

# 读取train数据
trainData = pd.read_table(
    './rawdata/IJCAI_18/round2_train.txt', sep=' ')

allData = trainData.append(testData)
del testData
del trainData
gc.collect()


# 将unix时间转换为正常的时间
allData['context_timestamp'] = pd.to_datetime(
    allData['context_timestamp'], unit='s')

# 给所有的sample的时间都引入一个8小时的便宜，这样刚好就是8天，从0点到0点
allData.context_timestamp += pd.to_timedelta('8:00:00')

# 引入day，以及当前样本的所处小时
allData.insert(loc=0, column='day', value=allData.context_timestamp.dt.day)
allData.insert(loc=0, column='hour', value=allData.context_timestamp.dt.hour)

# 训练集的第一天是31号，将其转换成0方便后面进一步处理
allData.loc[allData.day == 31, 'day'] = 0


def getStringVal(s, num):
    # 用于提取字符串中的某一段
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


# 引入商品类别信息以及类别信息长度
allData.insert(loc=0, column='item_cat_id',
               value=allData['item_category_list'].apply(
                lambda x: getStringVal(x, 2)))
allData.insert(loc=0, column='item_cat_len',
               value=allData['item_category_list'].apply(
                lambda x: x.count(';')))

# 引入标志，表示样本所在时间是上午还是下午
allData.insert(loc=0, column='isTestTime', value=0)
allData.loc[allData.hour >= 12, 'isTestTime'] = 1

# 划分数据集，方便后面进一步的特征提取
data05 = allData[allData.day.isin(range(6))]
data6 = allData[allData.day == 6]
data7 = allData[allData.day == 7]

del allData
gc.collect()


def getColInfo(col, dataset, rate_col=False, nameAdd=''):

    '''
    getColInfo: 函数用于进行一些组合特征的点击数量，以及点击的转换率的计算

    arg:
        col: list, 包含需要组合的特征
        dataset: pd.DataFrame, 需要统计的数据集
        rate_col: bool, 标志位，控制是否统计点击的转换率
        nameAdd: str, 统计出来的特征的名字需要附加上的字符串

    return:
        pd.DataFrame, 统计出来的数据
    '''

    name = ''
    for i in col:
        name += i+'_'
    name += nameAdd

    # 统计点击次数
    table_info = dataset[col+['day']].groupby(
            by=col, as_index=False).count()

    if not rate_col:
        table_info = table_info.fillna(0)
        table_info = table_info.rename(columns={'day': name+'counts'})

        # 统计出来的点击次数除以当天的总的点击数，用以降低不同天对于点击次数的影响
        table_info[name+'counts'] /= dataset.shape[0]
    else:
        # 统计交易的次数
        table_1_info = dataset[dataset['is_trade'] == 1][
            col+['day']].groupby(by=col, as_index=False).count()

        # 合并点击次数，以及交易次数
        table_info = pd.merge(left=table_info, right=table_1_info, how='outer',
                              on=col)

        table_info = table_info.fillna(0)
        table_info.insert(loc=0, column=name+'rate', value=0)

        # 用交易次数除以点击次数计算得到转换率
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
    '''
    mergeInfo: 将统计出来的数据合并到rawData

    arg:
        rawData: pd.DataFrame, 需要合并特征的原始数据
        datasetlist: list, 统计出来的特征数据集的list

    return:
        pd.DataFrame, 合并特征结束后的数据集
    '''
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

    '''
    mergeIdRankInfo: 提取点击顺序以及第一次点击到最后一次点击的时间差

    arg:
        dataset: pd.DataFrame, 需要提取特征的原始数据
        collist: list, 需要统计的特征的名字的list

    return:
        pd.DataFrame, 合并特征结束后的数据集
    '''

    for col in collist:
        if type(col) == str:
            print('processing:\t', col)
            # 将数据按照时间顺序排列
            dataset = dataset.sort_values(by=['context_timestamp'])
            # 提取出点击的顺序
            tempG = dataset.groupby(by=[col], as_index=False)
            dataset[col+'_rank'] = tempG.cumcount()
            # 将提取出来的顺序归一化
            dataset[col+'_rank'] /= (dataset[col+'_counts'] * dataset.shape[0])
            # 计算第一次点击到最后一次点击的时间差
            tempTimeperiod = \
                pd.DataFrame(
                    (tempG['context_timestamp'].last()['context_timestamp'] -
                     tempG['context_timestamp'].first()['context_timestamp']
                     ).dt.seconds)

            tempTimeperiod[col] = tempG['context_timestamp'].last()[col]
            tempTimeperiod = tempTimeperiod.rename(
                    {'context_timestamp': col+'_timeperiod'}, axis=1)

            # 将时间差合并到数据集上面
            dataset = pd.merge(dataset, tempTimeperiod, on=[col])

    return dataset


def getColStdInfo(col, dataset, nameAdd=''):
    '''
    getColStdInfo: 用于得到一些特征的方差，例如点击数量的方差
    '''
    dayPeriod = dataset.day.unique()
    info_list = []
#     print(dayPeriod)
    for day in dayPeriod:
        info_list.append(getColInfo(
            col, dataset[dataset.day == day],
            rate_col=True, nameAdd=str(day)+'_'))

    allinfo = pd.DataFrame(info_list[0])
    for item in info_list[1:]:
        allinfo = pd.merge(allinfo, item, how='outer', on=col)

    allinfo = allinfo.fillna(0)

    countsCollist = []
    rateCollist = []
    for feature in allinfo.columns:
        if 'counts' in feature:
            countsCollist.append(feature)
        if 'rate' in feature:
            rateCollist.append(feature)

    name = ''
    for i in col:
        name += i+'_'

    allinfo[name+'counts_std'+nameAdd] = allinfo[countsCollist].std(axis=1)
    allinfo[name+'counts_rate'+nameAdd] = allinfo[rateCollist].std(axis=1)
    allinfo = allinfo[
        col+[name+'counts_std'+nameAdd, name+'counts_rate'+nameAdd]]

#     print(allinfo.head(3))

    gc.collect()
    return allinfo


info_list = []
# =============================================================================
# 生成头5天的特征
# =============================================================================
collist = ['item_brand_id', 'item_id', 'item_city_id', 'user_id',
           'shop_id',


           ['user_id', 'item_id'],
           ['user_id', 'shop_id'],
           ['user_id', 'item_brand_id'],


           ['user_id', 'context_page_id'],
           ['user_id', 'item_cat_id'],
           ['user_id', 'item_price_level'],
           ['user_id', 'item_sales_level'],
           ['user_id', 'item_collected_level'],
           ['user_id', 'item_pv_level'],
           ['user_id', 'shop_star_level'],
           ['user_id', 'hour'],


           ['shop_id', 'context_page_id'],
           ['shop_id', 'item_cat_id'],
           ['shop_id', 'item_price_level'],
           ['shop_id', 'item_sales_level'],
           ['shop_id', 'item_collected_level'],
           ['shop_id', 'item_pv_level'],

           ['shop_id', 'user_star_level'],
           ['shop_id', 'user_age_level'],
           ['shop_id', 'user_occupation_id'],
           ['shop_id', 'user_gender_id'],

           ['shop_id', 'hour'],


           ['item_id', 'user_age_level'],
           ['item_id', 'user_star_level'],
           ['item_id', 'user_age_level'],
           ['item_id', 'user_occupation_id'],
           ['item_id', 'user_gender_id'],
           ['item_id', 'hour'],


           ['item_city_id', 'item_cat_id'],
           ['item_city_id', 'item_brand_id'],
           ['item_city_id', 'item_price_level'],

           ]

for col in collist:
    print(col)
    if type(col) == str:
        info_list.append(getColInfo([col], data05,
                                    rate_col=True, nameAdd='05_'))
#         info_list.append(getColStdInfo([col], data05, nameAdd='_05'))

    elif type(col) == list:
        info_list.append(getColInfo(col, data05,
                                    rate_col=True, nameAdd='05_'))
#         info_list.append(getColStdInfo(col, data05, nameAdd='_05'))


del data05
gc.collect()
# =============================================================================
# 生成第6天的特征
# =============================================================================
collist = ['item_brand_id', 'item_id', 'item_city_id', 'user_id',
           'shop_id',


           ['user_id', 'item_id'],
           ['user_id', 'shop_id'],
           ['user_id', 'item_brand_id'],


           ['user_id', 'context_page_id'],
           ['user_id', 'item_cat_id'],
           ['user_id', 'item_price_level'],
           ['user_id', 'item_sales_level'],
           ['user_id', 'item_collected_level'],
           ['user_id', 'item_pv_level'],
           ['user_id', 'shop_star_level'],
           ['user_id', 'hour'],


           ['shop_id', 'context_page_id'],
           ['shop_id', 'item_cat_id'],
           ['shop_id', 'item_price_level'],
           ['shop_id', 'item_sales_level'],
           ['shop_id', 'item_collected_level'],
           ['shop_id', 'item_pv_level'],

           ['shop_id', 'user_star_level'],
           ['shop_id', 'user_age_level'],
           ['shop_id', 'user_occupation_id'],
           ['shop_id', 'user_gender_id'],

           ['shop_id', 'hour'],


           ['item_id', 'user_age_level'],
           ['item_id', 'user_star_level'],
           ['item_id', 'user_age_level'],
           ['item_id', 'user_occupation_id'],
           ['item_id', 'user_gender_id'],
           ['item_id', 'hour'],


           ['item_city_id', 'item_cat_id'],
           ['item_city_id', 'item_brand_id'],
           ['item_city_id', 'item_price_level'],

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
           'shop_id',


           ['user_id', 'item_id'],
           ['user_id', 'shop_id'],
           ['user_id', 'item_brand_id'],


           ['user_id', 'context_page_id'],
           ['user_id', 'item_cat_id'],
           ['user_id', 'item_price_level'],
           ['user_id', 'item_sales_level'],
           ['user_id', 'item_collected_level'],
           ['user_id', 'item_pv_level'],
           ['user_id', 'shop_star_level'],
           ['user_id', 'hour'],


           ['shop_id', 'context_page_id'],
           ['shop_id', 'item_cat_id'],
           ['shop_id', 'item_price_level'],
           ['shop_id', 'item_sales_level'],
           ['shop_id', 'item_collected_level'],
           ['shop_id', 'item_pv_level'],

           ['shop_id', 'user_star_level'],
           ['shop_id', 'user_age_level'],
           ['shop_id', 'user_occupation_id'],
           ['shop_id', 'user_gender_id'],

           ['shop_id', 'hour'],


           ['item_id', 'user_age_level'],
           ['item_id', 'user_star_level'],
           ['item_id', 'user_age_level'],
           ['item_id', 'user_occupation_id'],
           ['item_id', 'user_gender_id'],
           ['item_id', 'hour'],


           ['item_city_id', 'item_cat_id'],
           ['item_city_id', 'item_brand_id'],
           ['item_city_id', 'item_price_level'],

           ]

for col in collist:
    if type(col) == str:
        info_list.append(getColInfo([col], data7))
    elif type(col) == list:
        info_list.append(getColInfo(col, data7))

colRanklist = ['user_id', 'item_id', 'item_brand_id', 'shop_id']

# 将提取的特征合并到第7天的数据集上面
data7 = mergeInfo(data7, info_list)
del info_list
gc.collect()
# 提取顺序相关的特征
data7 = mergeIdRankInfo(data7, colRanklist)
# 生成csv
data7.to_csv('./ProcessedData/day.csv', index=False)
gc.collect()
# =============================================================================
# 这个模块的作用是把上面的那些day.csv转换成模型训练用的csv
# =============================================================================

# data7 = pd.read_csv('./ProcessedData/day.csv')
# data7 = data7[data7.hour.isin(range(8,24))]
catFeatureslist = ['item_id', 'item_brand_id', 'item_city_id',
                   'user_id', 'user_gender_id', 'user_occupation_id',
                   'context_id', 'shop_id', 'item_cat_id',
                   ]

catDropList = [
               'context_id',
               'hour'
               ]


# 删除模型不能处理的特征
data7 = data7.drop(
    ['context_timestamp',
     'item_category_list',
     'item_property_list', 'predict_category_property'], axis=1)


# 删除没有意义的特征
data7 = data7.drop(catDropList, axis=1)

# 将类别特征转换成cat形式
for col in data7.columns:
    if col in catFeatureslist:
        data7[col] = pd.Categorical(data7[col])

# 将特征都归一化
for col in data7:
    print(col)
    if col not in catFeatureslist + ['instance_id']:
        if col not in ['day', 'is_trade', 'isTestTime']:
            minval = data7[col].min()
            maxval = data7[col].max()
            if maxval != minval:
                data7[col] = data7[col].apply(
                    lambda x: (x-minval)/(maxval-minval))
            else:
                data7 = data7.drop([col], axis=1)

# 划分数据集为上半天和下半天，也就是划分训练集以及测试集
train = data7[data7.isTestTime == 0].drop(
    ['day', 'instance_id', 'isTestTime'], axis=1)
test = data7[data7.isTestTime == 1].drop(
    ['is_trade', 'day',  'isTestTime'], axis=1)

# 生成csv
train.to_csv('./ProcessedData/train.csv', index=False)
test.to_csv('./ProcessedData/test.csv', index=False)

gc.collect()
