import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
from helpers.Data_prep import *
from helpers.Eda import *
from helpers.Models import *

# Veri setinin okutulması
df = pd.read_csv("Unit_1_Project_Dataset.csv")

######################################
# EDA
######################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


######################################
# Genel Resim
######################################

check_df(df)


################################
# Numerik ve Kategorik Değişkenlerin Yakalanması
##################################

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)


######################################
# Kategorik Değişken Analizi
######################################

for col in cat_cols:
    cat_summary(df, col)

######################################
# Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

for col in num_cols:
    num_summary(df, col)

######################################
# Hedef Değişken Analizi - KAtegorical (Analysis of Target Variable)
######################################

for col in cat_cols:
    target_summary_with_cat(df,"price",col)


######################################
# Korelasyon Analizi (Analysis of Correlation)
######################################

# Korelasyonların gösterilmesi
corr_matrix = df.corr()
threshold = 0.3
filtre = np.abs(corr_matrix["price"]) > threshold
corr_features=corr_matrix.columns[filtre].tolist()
sns.clustermap (df[corr_features].corr(), annot = True, fmt = ".2f",cmap = "plasma")
plt.title("Correlation Between Features")
plt.show()

##############################################################
# Data Preprocessing & Feature Engineering
##############################################################
# This part consists of 4 steps which are below:
# 1. Missing Values
# 2. Outliers
# 3. Rare Encoding, Label Encoding, One-Hot Encoding
# 4. Feature Scaling

######################################
# 1. Missing Value Analysis
######################################
missing_values_table(df)

#Missing Value'ları düzelt
df=MissingValueHandle(df)

##################################################################
#Outlier Detection
################################################################
df.drop(df[df['price']>7000].index,inplace=True)
##########################################################
#Base Model
###################################################
useless_columns=['host_id','host_name','id','latitude','longitude','zipcode','country','city','state','host_since_anniversary','host_since_year']
df.drop(useless_columns, axis = 1,inplace=True)

#Get Column Types
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
num_cols.remove("review_scores_cleanliness")
cat_cols.append("review_scores_cleanliness")
cat_but_car.remove("neighbourhood_cleansed")
cat_cols.append("neighbourhood_cleansed")

#Encoding
df_encode=one_hot_encoder(df,cat_cols)

#Scaling
num_cols = [col for col in num_cols if "price" not in col]

df_encode[num_cols]=RobustScaling(df_encode[num_cols],num_cols)

#BASE MODEL

PrepModel(df_encode,'price')

X = df_encode.drop('price', axis=1)
y = df_encode['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=112)
model=LGBMRegressor(n_estimators= 500, max_depth=8, learning_rate= 0.01)
hyper_parameter_model=model.fit(X_train, y_train)
#train hata
y_train_pred=hyper_parameter_model.predict(X_train)
print( np.sqrt(mean_squared_error(y_train, y_train_pred)))
#test hata
y_test_pred=hyper_parameter_model.predict(X_test)
print( np.sqrt(mean_squared_error(y_test, y_test_pred)))


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.show(Block=True)









