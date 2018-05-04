import pandas as pd


newData = pd.read_csv('./ProcessedData/day7.csv')
oldData = pd.read_csv('./IJCAI_ROUND2_TempData/day7.csv')

newData.isTesttime.unique()
newData = newData[newData.isTestTime == 0]



newData = newData.sort_values(by=['instance_id'])
oldData = oldData.sort_values(by=['instance_id'])

newData.head()


newData.head()
oldData.head()
for col in newData.columns:
    try:
        print(col,'\t',newData[col].std(),'\t',oldData[col].std())
        print(col,'\t',newData[col].std(),'\t',oldData[col].std())
        print()
    except(Exception):
        print(col)


newData.user_id_rank.head()
oldData.user_id_rank.head()


newData.user_id_rank.max()
newData.user_id_rank.min()
newData.user_id_rank.mean()








newData.head()
oldData.user_id_rank.max()
oldData.user_id_rank.min()
oldData.user_id_rank.mean()



newData.user_id_difftime_y.max()
newData.user_id_difftime_y.min()
newData.user_id_difftime_y.mean()


newData.user_id_timeperiod.max()
newData.user_id_timeperiod.mean()
newData.user_id_timeperiod.min()

oldData.user_id_timeperiod.max()
oldData.user_id_timeperiod.mean()
oldData.user_id_timeperiod.min()

(oldData.user_id_counts * oldData.shape[0]).max()
(oldData.user_id_counts * oldData.shape[0]).min()
(oldData.user_id_counts * oldData.shape[0]).mean()


(oldData.user_id_rank * (oldData.user_id_counts * oldData.shape[0])).max()
(oldData.user_id_rank * (oldData.user_id_counts * oldData.shape[0])).min()
(oldData.user_id_rank * (oldData.user_id_counts * oldData.shape[0])).mean()
