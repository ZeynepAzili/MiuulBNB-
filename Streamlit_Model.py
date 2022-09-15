
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import warnings
from sklearn.exceptions import ConvergenceWarning


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
from helpers.Data_prep import *
from helpers.Eda import *
from helpers.Models import *

df=pd.read_csv('Unit_1_Project_Dataset.csv')

##########################################################
#feature engineering
###################################################
df=FeatureEngineering(df)

#missing value
df=MissingValueHandle(df)

#Outlier detection
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
num_cols.remove("review_scores_cleanliness")
cat_cols.append("review_scores_cleanliness")

#Check outliers

df.drop(df[df['price']>300].index,inplace=True)
for col in num_cols:
    replace_with_thresholds(df, col)

df_scores=LOFOutlierDetection(df,20,num_cols)
th = np.sort(df_scores)[0:10][2]
df.drop(df[df_scores < th].index, inplace=True)

#Kullanılacak kolonlar
model_columns=['NEW_person_By_Area','NEW_DISTRICT','room_type','property_type','accommodates','guests_included','extra_people'
    ,'bedrooms','bathrooms','review_scores_checkin','review_scores_location','review_scores_accuracy','review_scores_communication','price']

df=df[model_columns]
cat_cols=['room_type','property_type','NEW_DISTRICT','review_scores_checkin','review_scores_location'
    ,'review_scores_communication','review_scores_accuracy']
num_cols = [col for col in model_columns if col not in ['room_type','property_type','NEW_DISTRICT',
            'review_scores_checkin','review_scores_location','review_scores_communication','review_scores_accuracy'] ]
#Encoding
df_encode=one_hot_encoder(df,cat_cols)

#Scaling
num_cols = [col for col in num_cols if "price" not in col]
df_encode[num_cols]=RobustScaling(df_encode[num_cols],num_cols)

X = df_encode.drop('price', axis=1)
y = df_encode['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=112)
alphas= 10**np.linspace(10,-2,100)*0.5
lasso_cv = RidgeCV(alphas = alphas, cv = 10)
lasso_cv_model = lasso_cv.fit(X,y)

ls = Ridge(lasso_cv_model.alpha_)
lasso_tuned_model = ls.fit(X_train,y_train)

import pickle
pickle.dump(lasso_tuned_model,open('Airbnb2.pckl','wb'))


