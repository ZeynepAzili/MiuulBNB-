import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV,RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import VotingRegressor


def PrepModel(Dataframe,Dependent):
    X = Dataframe.drop(Dependent, axis=1)
    y = Dataframe[Dependent]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=112)

    models = [('LR', LinearRegression()),
          ('Lasso', Lasso()),
          ('Ridge', Ridge()),
          ('ElasticNet', ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGB", XGBRegressor()),
          ("LGBM", LGBMRegressor())
          # ("CatBoost", CatBoostRegressor(verbose=False))
          ]
    for name, regressor in models:
        modell = regressor.fit(X_train, y_train)
        y_pred_train = modell.predict(X_train)
        print(name + "Train RMSE Değeri: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
        y_pred_test=modell.predict(X_test)
        print(name + "Test RMSE Değeri: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(name, f"RMSE crossval Değeri: {round(rmse, 4)} ({name}) ")


#Hyper parameter tunning


def hyperparameter_optimization(X, y, cv=10, scoring="neg_mean_squared_error"):
    rf_params = {"max_depth": [3, 5, 8, 15, None],
                 "max_features": [5, 7, 10, 15, "auto","log2","sqrt"],
                 "min_samples_split": [2, 15, 20, 25],
                 "n_estimators": [10, 50, 100, 200, 300],
                 "min_samples_leaf":[1,2,3,5,8]}

    xgboost_params = {"eta": [0.3, 0.01, 0.05, 0.2],
                      "max_depth": [6, 8, 10],
                      "colsample_bytree": [0.5, 1]}

    lightgbm_params = {"learning_rate": [0.1, 0.08, 0.05, 0.01],
                       "max_depth": [-1, 5, 8, 10],
                       "n_estimators": [100, 200, 300, 500]}

    gbm_params = {"learning_rate": [0.1, 0.08, 0.05, 0.01],
                  "min_samples_split": [2, 5, 8],
                  "n_estimators": [100, 200, 300, 500],
                  "max_depth": [3, 5, 8],
                  'loss': ['squared_error']}

    regressors = [("RF", RandomForestRegressor(), rf_params),
                # ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
                #  ('LightGBM', LGBMRegressor(), lightgbm_params),
                #  ('GBM', GradientBoostingRegressor(), gbm_params)
    ]

    print("Hyperparameter Optimization....")
    best_models = {}

    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring=scoring)))
        print(f"{scoring} (Before): {round(cv_results, 4)}")

        gs_best = RandomizedSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=cv, scoring=scoring)))
        print(f"{scoring} (After): {round(cv_results, 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def voting_classifier(best_models,cv=10, X='', y=''):
    print("Voting Regressor...")

    voting_clf = VotingRegressor(estimators=[('XGBoost', best_models["XGBoost"]),
                                              ('GBM', best_models["GBM"]),
                                              ('LightGBM', best_models["LightGBM"])]).fit(X, y)

    cv_results =  np.mean(np.sqrt(-cross_val_score(voting_clf, X, y, cv=cv, scoring='neg_mean_squared_error')))
    print(f"(Results): {round(cv_results, 4)}")
    return voting_clf


def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[0:num]) # num argümanı ile barplot üzerinde kaç değişkeni göstereceğimizi seçebiliriz.
    plt.title("Features")
    plt.title(f"Features for {type(model).__name__}") # modelin __name__ metodu ile adını döndürür
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')