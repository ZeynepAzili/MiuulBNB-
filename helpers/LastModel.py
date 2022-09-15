import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
from sklearn.linear_model import ElasticNet,ElasticNetCV
import warnings
from sklearn.exceptions import ConvergenceWarning


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
from helpers.Data_prep import *
from helpers.Eda import *
from helpers.Models import *

###################################################
##MODELİ GELİŞTİRİYORUZ
##################################################
df=pd.read_csv('Unit_1_Project_Dataset.csv')

##########################################################
#feature engineering
###################################################
df=FeatureEngineering(df)

#missing value
df=MissingValueHandle(df)

##################################################################
#Outlier Detection
################################################################
#Tespit 1

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
num_cols.remove("review_scores_cleanliness")
cat_cols.append("review_scores_cleanliness")

#Check outliers
for col in num_cols:
    grab_outliers(df,col)

df.drop(df[df['price']>300].index,inplace=True)

for col in num_cols:
    replace_with_thresholds(df, col)

#LOF Outlier
df_scores=LOFOutlierDetection(df,20,num_cols)
th = np.sort(df_scores)[0:10][2]
df.drop(df[df_scores < th].index, inplace=True)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

#Rare encoder

#Analyze Rare Variables
#rare_analyser(df, 'price', cat_cols)

#propertytype 0.01 ten küçükler
#bed_type datadan çıkabilir
#review olanlar hem rare hem değil denenebilir.
#neighbourhood_cleansed datadan çıkabilir.
"""
rare_columns=[col for col in cat_cols if col in['review_scores_accuracy','review_scores_cleanliness','review_scores_checkin'
            ,'review_scores_communication','review_scores_location','review_scores_value']]
df = rare_encoder(df, 0.015, rare_columns)
"""
#Yapılan inceleme sonuçlarına göre rare analyzer uygulamayacağız.

#Correlation Analysis

corr_matrix = df[num_cols].corr()
sns.clustermap (corr_matrix, annot = True, fmt = ".2f",cmap = "viridis")
plt.title("Correlation Between Features w/ Corr Threshold 0.75")
plt.show(block=True)
#acommodates,beds,bedrooms,review_scores_value
##########################################################
#Ana Model Model
###################################################
useless_columns=['host_id','host_name','id','latitude','longitude','zipcode','country','city','state','review_scores_rating','beds','neighbourhood_cleansed'
                 ,'bed_type']
df.drop(useless_columns, axis = 1,inplace=True)
#df.columns

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
num_cols.remove("review_scores_cleanliness")
cat_cols.append("review_scores_cleanliness")


#Encoding
df_encode=one_hot_encoder(df,cat_cols)

#Scaling
#import numpy as np
#df_encode['price'] = np.log(df_encode['price'])

num_cols = [col for col in num_cols if "price" not in col]
df_encode[num_cols]=RobustScaling(df_encode[num_cols],num_cols)

###############################
#Model
##################################
PrepModel(df_encode,'price')

#Modelden Sonra Correlation Analysis
corr_matrix = df[num_cols].corr()
sns.clustermap (corr_matrix, annot = True, fmt = ".2f",cmap = "viridis")
plt.title("Correlation Between Features w/ Corr Threshold 0.75")
plt.show(Block=True)


#################################################################################
#Hyperparameter tunning
###########################################################

#GBM
X = df_encode.drop('price', axis=1)
y = df_encode['price']


best_models = hyperparameter_optimization(X, y)

#Aşağıdaki 4 model için test edilen optimum değerler:
#RF=>n_estimators= 200, max_depth=8, min_samples_split= 15, max_features='auto',min_samples_leaf=8

#XGBoost=> subsample= 0.5, max_depth=6, min_child_weight= 1, gamma=1,eta=0.2,colsample_bytree=0.5,alpha=0

#LGBM=>#path_smooth= 0.02, num_leaves= 100, n_estimators= 100, min_sum_hessian_in_leaf= 15,
#min_gain_to_split= 0.1, max_depth= 8, max_bin= 20, learning_rate= 0.05, lambda_l2= 0.5, lambda_l1= 0.5,
#extra_trees= True, bagging_freq= 10, bagging_fraction= 0.5

#GBM=> subsample= 0.8, n_estimators= 300, min_samples_split= 200, min_samples_leaf= 20, \
#max_features= 'log2', max_depth= 8, loss= 'absolute_error', learning_rate= 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=112)

model=GradientBoostingRegressor(subsample= 0.8, n_estimators= 300, min_samples_split= 200, min_samples_leaf= 20,
max_features= 'log2', max_depth= 8, loss= 'absolute_error', learning_rate= 0.1)
hyper_parameter_model=model.fit(X_train, y_train)
#train hata
y_train_pred=hyper_parameter_model.predict(X_train)
print( np.sqrt(mean_squared_error(y_train, y_train_pred)))#45

#test hata
y_test_pred=hyper_parameter_model.predict(X_test)
print( np.sqrt(mean_squared_error(y_test, y_test_pred)))#44
#Cross Val
cv_results = np.mean(np.sqrt(-cross_val_score(hyper_parameter_model, X, y, cv=10, scoring='neg_mean_squared_error')))
print(f"(Results): {round(cv_results, 4)}")

#Lasso-Ridge-Elastic Net Tuning

X = df_encode.drop('price', axis=1)
y = df_encode['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=112)
alphas= 10**np.linspace(10,-2,100)*0.5
lasso_cv = RidgeCV(alphas = alphas, cv = 10)
lasso_cv_model = lasso_cv.fit(X,y)
lasso_cv_model.alpha_

ls = Ridge(lasso_cv_model.alpha_)
lasso_tuned_model = ls.fit(X_test,y_test)
y_pred = lasso_tuned_model.predict(X_test)
print("RMSE:" , np.sqrt(mean_squared_error(y_test, y_pred))) # 202.661

cv_results = np.mean(np.sqrt(-cross_val_score(lasso_tuned_model, X, y, cv=10, scoring='neg_mean_squared_error')))
print(f"(Results): {round(cv_results, 4)}")


#Ridge feature importance

# Ridge Regression Coefficients: Katsayılardan önem düzeyi düşük değişkenleri tespit edelim
Importance = pd.DataFrame({"Feature": X.columns,
                           "Coefs" : lasso_tuned_model.coef_ })

Importance.sort_values("Coefs").sort_values("Coefs", ascending=True)["Coefs"]

# Katsayısı >0 olanları seçelim:
selected_features = list(Importance[Importance["Coefs"] > 0]["Feature"])

X=Importance.sort_values("Coefs").sort_values("Coefs", ascending=False)
X=X[np.abs(X['Coefs'])>4]

import seaborn as sns
sns.barplot(x="Coefs", y="Feature", data=X,label="Alcohol-involved")
plt.show(block=True)
df.columns

"""
> Sonuç açısından fark yaratmadı.
#VotingRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=112)
Rf_model=RandomForestRegressor(n_estimators= 200, max_depth=8, min_samples_split= 15, max_features='auto',min_samples_leaf=8)
GBM_model=GradientBoostingRegressor(subsample= 0.8, n_estimators= 300, min_samples_split= 200, min_samples_leaf= 20, \
max_features= 'log2', max_depth= 8, loss= 'absolute_error', learning_rate= 0.1)
LGBM_Model=LGBMRegressor(path_smooth= 0.02, num_leaves= 100, n_estimators= 100, min_sum_hessian_in_leaf= 15,
min_gain_to_split= 0.1, max_depth= 8, max_bin= 20, learning_rate= 0.05, lambda_l2= 0.5, lambda_l1= 0.5,
extra_trees= True, bagging_freq= 10, bagging_fraction= 0.5)
XGBoost_Model=XGBRegressor(subsample= 0.5, max_depth=6, min_child_weight= 1, gamma=1,eta=0.2,colsample_bytree=0.5,alpha=0)


voting_clf = VotingRegressor(estimators=[#('RF', Rf_model),
#('XGBoost', XGBoost_Model),
('GBM', GBM_model),
('LightGBM', LGBM_Model)]).fit(X_test, y_test)

y_train_pred=voting_clf.predict(X_train)
print( np.sqrt(mean_squared_error(y_train, y_train_pred)))#45
#Cross Val
cv_results = np.mean(np.sqrt(-cross_val_score(voting_clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error')))
print(f"(Results): {round(cv_results, 4)}")

#test hata
y_test_pred=voting_clf.predict(X_test)
print( np.sqrt(mean_squared_error(y_test, y_test_pred)))#44

#Cross Val
cv_results = np.mean(np.sqrt(-cross_val_score(voting_clf, X_test, y_test, cv=10, scoring='neg_mean_squared_error')))
print(f"(Results): {round(cv_results, 4)}")
"""



#Train test görsel

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.show(block=True)


#sns.regplot(x=y_test,y=y_test_pred,ci=None,color ='red');
sns.regplot(x=y_train,y=y_train_pred,ci=None,color ='red');
plt.show(block=True)


