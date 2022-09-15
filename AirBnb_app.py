import pickle

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from streamlit_folium import st_folium
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

from helpers.Data_prep import *
from helpers.Eda import *
from helpers.Models import *

st.set_page_config(layout='wide',initial_sidebar_state ='expanded',page_title="MIULBNB!",page_icon="🏡")

#SIDEBAR
st.sidebar.header('USER INPUT FEATURES')
def user_input_features():
    Accommodates=st.sidebar.number_input(label='Host Capacity',min_value=1,max_value=8,value=3)
    guests_included = st.sidebar.number_input(label='Number of Guest', min_value=0, max_value=5, value=2)
    ExtraPeople = st.sidebar.slider('How much Euros Each Extra Person?', 0, 100, 0)
    Bedrooms = st.sidebar.number_input(label='Bedroom Count', min_value=1, max_value=10, value=2)
    Bathrooms=st.sidebar.number_input(label='Bathrooms Count', min_value=0, max_value=3, value=1)
    RoomType = st.sidebar.radio('Room Type Selection', ('Entire home/apt', 'Private room','Shared room'))
    PropertyType = st.sidebar.selectbox('Property Type Selection', ('Apartment', 'House', 'Bed & Breakfast','Boat','Loft','Other','Cabin','Camper/RV',                                                            'Villa'))
    Neighbourhood = st.sidebar.selectbox('Neighbourhood Selection', ('Centrum-West', 'De Baarsjes - Oud-West','Centrum-Oost','De Pijp - Rivierenbuurt',
    'Westerpark','Zuid','Oud-Oost','Bos en Lommer','Oud-Noord','Watergraafsmeer','Slotervaart','IJburg - Zeeburgereiland','Buitenveldert - Zuidas',
    'Noord-West','Geuzenveld - Slotermeer','Noord-Oost','Osdorp','De Aker - Nieuw Sloten','Bijlmer-Centrum','Bijlmer-Oost','Gaasperdam - Driemond'))
    ReviewScoreCheckin = st.sidebar.number_input('Checkin Review Score', min_value=0, max_value=10, value=0)
    ReviewScoreLocation = st.sidebar.number_input('Location Review Score',min_value=0, max_value=10, value=0)
    ReviewScoreCommunication = st.sidebar.number_input('Communication Review Score', min_value=0, max_value=30, value=0)
    ReviewScoreAccuracy = st.sidebar.number_input('Accuracy Review Score', min_value=0, max_value=10, value=0)

    data = {'Accommodates': [Accommodates],
            'guests_included': [guests_included],
            'ExtraPeople': [ExtraPeople],
            'Bedrooms': [Bedrooms],
            'Bathrooms': [Bathrooms],
            'RoomType': [RoomType],
            'PropertyType': [PropertyType],
            'Neighbourhood': [Neighbourhood],
            'ReviewScoreCheckin': [ReviewScoreCheckin],
            'ReviewScoreLocation': [ReviewScoreLocation],
            'ReviewScoreCommunication': [ReviewScoreCommunication],
            'ReviewScoreAccuracy': [ReviewScoreAccuracy]
            }

    features = pd.DataFrame(data,index=[0])

    return features

#Sayfaya Resim ve Başlık Ekle
image = Image.open('WhatsApp Image 2022-06-17 at 21.07.16.jpeg')
st.image(image,width=800)
st.header('MIUULBNB PRICE PREDICTION APP')

#Seçilen Değerleri DF yapıp göster
input_df = user_input_features()
st.header('User Choices')
st.write(input_df)

#Girilen değerler için df oluştur
#New Person By Area Oluştur
df=pd.read_csv('Unit_1_Project_Dataset.csv')
df['zipcode'] = df['zipcode'].str[0:4]
neighbourhood_cleansed_mode = df.groupby(['neighbourhood_cleansed'])['zipcode'].agg(pd.Series.mode).reset_index()
input_df['zip_code']=neighbourhood_cleansed_mode[neighbourhood_cleansed_mode['neighbourhood_cleansed']==input_df['Neighbourhood'][0]]['zipcode'].values

DataFrame_detay = pd.read_csv("Amsterdam_nufus.csv", sep=";", encoding='unicode_escape')
df['zipcode'] = df['zipcode'].astype(str)
DataFrame_detay['zipcode'] = DataFrame_detay['zipcode'].astype(str)

input_df['Population'] = (DataFrame_detay[DataFrame_detay['zipcode']==input_df['zip_code'][0]]['Population']).values
input_df['Area'] = (DataFrame_detay[DataFrame_detay['zipcode']==input_df['zip_code'][0]]['Area']).values
input_df['NEW_person_By_Area']=input_df['Population']/input_df['Area']
input_df.drop(['zip_code','Population','Area'],axis=1,inplace=True)

#NEW_DISTRICT oluştur.
input_df.loc[input_df['Neighbourhood'] == 'Centrum-West', 'NEW_DISTRICT'] = 'Center'
input_df.loc[input_df['Neighbourhood'] == 'De Baarsjes - Oud-West', 'NEW_DISTRICT'] = 'West'
input_df.loc[input_df['Neighbourhood'] == 'Centrum-Oost', 'NEW_DISTRICT'] = 'Center'
input_df.loc[input_df['Neighbourhood'] == 'De Pijp - Rivierenbuurt', 'NEW_DISTRICT'] = 'Zuid'
input_df.loc[input_df['Neighbourhood'] == 'Westerpark', 'NEW_DISTRICT'] = 'West'
input_df.loc[input_df['Neighbourhood'] == 'Zuid', 'NEW_DISTRICT'] = 'Zuid'
input_df.loc[input_df['Neighbourhood'] == 'Oud-Oost', 'NEW_DISTRICT'] = 'Oost'
input_df.loc[input_df['Neighbourhood'] == 'Bos en Lommer', 'NEW_DISTRICT'] = 'West'
input_df.loc[input_df['Neighbourhood'] == 'Oostelijk Havengebied - Indische Buurt', 'NEW_DISTRICT'] = 'Oost'
input_df.loc[input_df['Neighbourhood'] == 'Oud-Noord', 'NEW_DISTRICT'] = 'Noord'
input_df.loc[input_df['Neighbourhood'] == 'Watergraafsmeer', 'NEW_DISTRICT'] = 'Oost'
input_df.loc[input_df['Neighbourhood'] == 'Slotervaart', 'NEW_DISTRICT'] = 'Nieuw-West'
input_df.loc[input_df['Neighbourhood'] == 'Geuzenveld - Slotermeer', 'NEW_DISTRICT'] = 'Nieuw-West'
input_df.loc[input_df['Neighbourhood'] == 'De Aker - Nieuw Sloten', 'NEW_DISTRICT'] = 'Nieuw-West'
input_df.loc[input_df['Neighbourhood'] == 'Osdorp', 'NEW_DISTRICT'] = 'Nieuw-West'
input_df.loc[input_df['Neighbourhood'] == 'IJburg - Zeeburgereiland', 'NEW_DISTRICT'] = 'Oost'
input_df.loc[input_df['Neighbourhood'] == 'Buitenveldert - Zuidas', 'NEW_DISTRICT'] = 'Zuid'
input_df.loc[input_df['Neighbourhood'] == 'Noord-West', 'NEW_DISTRICT'] = 'Noord'
input_df.loc[input_df['Neighbourhood'] == 'Noord-Oost', 'NEW_DISTRICT'] = 'Noord'
input_df.loc[input_df['Neighbourhood'] == 'Bijlmer-Centrum', 'NEW_DISTRICT'] = 'Zuidoost'
input_df.loc[input_df['Neighbourhood'] == 'Bijlmer-Oost', 'NEW_DISTRICT'] = 'Zuidoost'
input_df.loc[input_df['Neighbourhood'] == 'Gaasperdam - Driemond', 'NEW_DISTRICT'] = 'Zuidoost'

input_df.drop(['Neighbourhood'],axis=1,inplace=True)

#Model kur
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

#Feature Engineering
df=FeatureEngineering(df)

#missing value
df=MissingValueHandle(df)

#Handle Outlier
df.drop(df[df['price']>300].index,inplace=True)
for col in num_cols:
    replace_with_thresholds(df, col)

df_scores=LOFOutlierDetection(df,20,num_cols)
th = np.sort(df_scores)[0:10][2]
df.drop(df[df_scores < th].index, inplace=True)


model_columns=['NEW_person_By_Area','NEW_DISTRICT','room_type','property_type','accommodates','guests_included','extra_people'
    ,'bedrooms','bathrooms','review_scores_checkin','review_scores_location','review_scores_accuracy','review_scores_communication']

#input data ile csv'yi birleştir.
input_df.columns=['accommodates','guests_included','extra_people','bedrooms','bathrooms','room_type','property_type',
                  'review_scores_checkin','review_scores_location','review_scores_communication','review_scores_accuracy',
                  'NEW_person_By_Area', 'NEW_DISTRICT'
                  ]
All_data=pd.concat([input_df,df[model_columns]],axis=0)

cat_cols=['room_type','property_type','NEW_DISTRICT','review_scores_checkin','review_scores_location','review_scores_communication','review_scores_accuracy']
num_cols = [col for col in input_df.columns if col not in ['price','room_type','property_type','NEW_DISTRICT',
                                                           'review_scores_checkin','review_scores_location',
                                                           'review_scores_communication','review_scores_accuracy'] ]
df_encode=one_hot_encoder(All_data,cat_cols)

df_encode[num_cols]=RobustScaling(df_encode[num_cols],num_cols)

df_encode=df_encode[:1]

len(df_encode.columns)

#Read pickle

load_model=pickle.load(open('Airbnb2.pckl','rb'))
prediction=load_model.predict(df_encode)

#import shap
#explainer=shap.TreeExplainer(load_model)
#shap_values=explainer.shap_values(df_encode)

st.header('Price Prediction Result')
prediction

#Map ekle

#plotting a map with the above defined points
#data=pd.read_csv('Unit_1_Project_Dataset.csv')

#st.header("Price Distribution in Amsterdam Neibourhood?")
# plot the slider that selects number of person died
#prices = st.slider("Price of Lots", int(data["price"].min()), int(data["price"].max()))
#bedds = st.slider("How Many Beds", int(data["beds"].min()), int(data["beds"].max()))
#st.map(data.query("price <= @prices & beds<=@bedds")[["latitude", "longitude"]].dropna(how ="any"))



