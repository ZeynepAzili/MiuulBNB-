import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import datetime as dt
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Thresholds values are retained and replaced with outlier values.
# Recovers from data loss in case of deletion.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# is there an outlier or not?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 0: # show top 5 if number of rows is greater than 10
        print(col_name,"  :  ",len(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]),' : ',dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(10))


# When outliers in a variable are deleted, other lines in that variable are deleted.
# Deleting an element in a variable can cause serious data loss
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def LOFOutlierDetection(DataFrame,n_neighbors,num_cols):
    df_ = DataFrame[num_cols]

    # LOF Scores:
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    clf.fit_predict(df_)
    df_scores = clf.negative_outlier_factor_

    # LOF Visualization:
    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 10], style='.-')
    plt.show(block=True)
    return df_scores


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe



def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) <= rare_perc).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), "Rare", dataframe[col])
    return dataframe

def RobustScaling(dataframe, col_name):
    rs = RobustScaler()
    dataframe[col_name] = rs.fit_transform(dataframe[col_name])
    return dataframe


def MissingValueHandle(dataframe):
    review_columns = dataframe.loc[:, (dataframe.columns.str.contains('review'))].columns
    dataframe[review_columns] = dataframe[review_columns].fillna(0)
    dataframe['host_response_time'] = dataframe['host_response_time'].fillna('None')
    dataframe['host_response_rate'] = dataframe['host_response_rate'].fillna(0)
    # State:%99'u north Holland. Useless column. Boşları North Holland yapabiliriz.
    dataframe['state'].fillna('North Holland', inplace=True)
    dataframe.dropna(inplace=True)
    return dataframe

def FeatureEngineering (DataFrame):
    #Öncelikle dublike kayıtları sil.
    # Dublike id'leri siliyoruz.
    x = DataFrame.groupby("id").agg({"id": "count"})
    x.columns = ['Cnt']
    x.reset_index(inplace=True)
    dublicated_id = x[x.Cnt > 1]['id']

    lst = []
    for col in dublicated_id:
        del_index = DataFrame[DataFrame['id'] == col].index.min()
        lst.append(del_index)

    DataFrame = DataFrame[~DataFrame.index.isin(lst)]

    # 1
    # host_since_year+host_since_anniversary ile yeni tarih türetilecek.ve bu değişkenler silinecek
    # df.drop('market',axis=1,inplace=True)
    DataFrame['marker'] = DataFrame['host_since_anniversary'].str.find('/')
    DataFrame.loc[DataFrame['marker'] == 1, 'host_since_anniversary'] = '0' + DataFrame[DataFrame['marker'] == 1]['host_since_anniversary']

    DataFrame['date_len'] = DataFrame['host_since_anniversary'].str.len()
    DataFrame['right_digit'] = DataFrame['host_since_anniversary'].str[-1:]
    DataFrame.loc[DataFrame['date_len'] == 4, 'host_since_anniversary'] = DataFrame[DataFrame['date_len'] == 4]['host_since_anniversary'].str[
                                                            0:3] + '0' + DataFrame['right_digit']
    DataFrame.drop(['date_len', 'right_digit', 'marker'], axis=1, inplace=True)

    DataFrame['host_since_date'] = pd.to_datetime(DataFrame['host_since_year'].astype(str) + '/' + DataFrame['host_since_anniversary'])
    DataFrame.drop(['host_since_year', 'host_since_anniversary'], axis=1, inplace=True)

    # Analizin yapıldığın gün 25.06.2015 olsun
    today_date = dt.datetime(2015, 6, 25)
    DataFrame['host_since_days'] = (today_date - DataFrame['host_since_date']).dt.days
    DataFrame.drop(['host_since_date'], axis=1, inplace=True)

    # host_since_days'i kategoriğe çevir
    night_bins = [81, 360, 588, 807, 1080, 2509]
    night_labels = ["81_360", "361_588", "589_807", "808_1080", "1081_2509"]

    DataFrame["NEW_Host_Start_Day_CAT"] = pd.cut(DataFrame["host_since_days"], bins=night_bins, labels=night_labels)
    DataFrame.drop(['host_since_days'], axis=1, inplace=True)

    #2
    #Neighbourhood bilgisine göre Amsterdam'ın 7 bölgesini ekleyebiliriz.
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Centrum-West', 'NEW_DISTRICT'] = 'Center'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'De Baarsjes - Oud-West', 'NEW_DISTRICT'] = 'West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Centrum-Oost', 'NEW_DISTRICT'] = 'Center'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'De Pijp - Rivierenbuurt', 'NEW_DISTRICT'] = 'Zuid'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Westerpark', 'NEW_DISTRICT'] = 'West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Zuid', 'NEW_DISTRICT'] = 'Zuid'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Oud-Oost', 'NEW_DISTRICT'] = 'Oost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Bos en Lommer', 'NEW_DISTRICT'] = 'West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Oostelijk Havengebied - Indische Buurt', 'NEW_DISTRICT'] = 'Oost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Oud-Noord', 'NEW_DISTRICT'] = 'Noord'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Watergraafsmeer', 'NEW_DISTRICT'] = 'Oost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Slotervaart', 'NEW_DISTRICT'] = 'Nieuw-West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Geuzenveld - Slotermeer', 'NEW_DISTRICT'] = 'Nieuw-West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'De Aker - Nieuw Sloten', 'NEW_DISTRICT'] = 'Nieuw-West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Osdorp', 'NEW_DISTRICT'] = 'Nieuw-West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'IJburg - Zeeburgereiland', 'NEW_DISTRICT'] = 'Oost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Buitenveldert - Zuidas', 'NEW_DISTRICT'] = 'Zuid'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Noord-West', 'NEW_DISTRICT'] = 'Noord'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Noord-Oost', 'NEW_DISTRICT'] = 'Noord'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Bijlmer-Centrum', 'NEW_DISTRICT'] = 'Zuidoost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Bijlmer-Oost', 'NEW_DISTRICT'] = 'Zuidoost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Gaasperdam - Driemond', 'NEW_DISTRICT'] = 'Zuidoost'

    #3
    #DataFrame['NEW_Acc_Multiply_Beds'] = DataFrame['accommodates'] * DataFrame['beds']

    #4
    DataFrame['NEW_Bed_Divide_Person'] = DataFrame['guests_included'] / DataFrame['beds']

    #5
    DataFrame['zipcode'] = DataFrame['zipcode'].str[0:4]
    neighbourhood_cleansed_mode = DataFrame.groupby(['neighbourhood_cleansed'])['zipcode'].agg(pd.Series.mode).reset_index()
    DataFrame = pd.merge(DataFrame, neighbourhood_cleansed_mode, how="left", on=["neighbourhood_cleansed"])
    DataFrame.loc[DataFrame['zipcode_x'].isnull(), 'zipcode_x'] = DataFrame[DataFrame['zipcode_x'].isnull()]['zipcode_y']
    DataFrame.drop('zipcode_y', inplace=True, axis=1)
    DataFrame.rename({'zipcode_x': 'zipcode'}, axis=1, inplace=True)

    #İnternetten bulunan veri ile merge edilir.
    DataFrame_detay = pd.read_csv("Amsterdam_nufus.csv", sep=";", encoding='unicode_escape')

    DataFrame['zipcode'] = DataFrame['zipcode'].astype(str)
    DataFrame_detay['zipcode'] = DataFrame_detay['zipcode'].astype(str)

    DataFrame = pd.merge(DataFrame, DataFrame_detay, how="left", on=["zipcode"])
    DataFrame['NEW_person_By_Area'] = DataFrame['Population'] / DataFrame['Area']
    DataFrame.drop(['Population','Area'],axis=1,inplace=True)

    return DataFrame
