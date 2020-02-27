
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
import numpy as np


# XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor
# Metrics : rmse, r2, mae

class Model_train():
    def __init__(self, data, name_target, type_of_dataset, name_of_dataset, name_of_model,
    seed=123, test_size=0.33, cv=5):
        super(Model_train, self).__init__()
        self.data = data
        self.name_target = name_target
        self.seed = seed
        self.test_size = test_size
        self.cv = cv
        self.type_of_dataset = type_of_dataset
        self.name_of_dataset = name_of_dataset
        self.name_of_model = name_of_model

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.data.copy().drop([self.name_target], axis=1),
        self.data.copy()[self.name_target].values, test_size=self.test_size, random_state=self.seed)

    def BRreg(self,n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, 
    compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
    
        model = linear_model.BayesianRidge(n_iter=n_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, 
        alpha_init=alpha_init, lambda_init=lambda_init, compute_score=compute_score, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
        verbose=verbose)
        kf = KFold(n_splits=self.cv)
        test_result = {'RMSE' : [], 'R2' : [], 'MAE' : []}
        for train_index, test_index in kf.split(self.data.copy()):
            train_X = self.data.copy().drop([self.name_target], axis=1).iloc[train_index, :]
            train_y = self.data.copy().loc[train_index, [self.name_target]]
            test_X = self.data.copy().drop([self.name_target], axis=1).iloc[test_index, :]
            test_y = self.data.copy().loc[test_index, [self.name_target]]
            model.fit(train_X, train_y)
            # 测试
            y_pred = model.predict(test_X)
            test_result['RMSE'].append(np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
            test_result['R2'].append(metrics.r2_score(test_y, y_pred))
            test_result['MAE'].append(metrics.mean_absolute_error(test_y, y_pred))

        for key, values in test_result.items():
            test_result[key] = np.array(values).mean()

        return model, test_result, train_X,test_X,test_y

    def Lreg(self, alpha=1, fit_intercept=True, normalize=False, copy_X=True,
         max_iter=300, tol=0.001, random_state=None, selection='cyclic'):

        model = linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, 
                                   max_iter=max_iter, tol=tol,random_state=random_state, selection=selection)
        kf = KFold(n_splits=self.cv)
        test_result = {'RMSE': [], 'R2': [], 'MAE': []}
        for train_index, test_index in kf.split(self.data.copy()):
            train_X = self.data.copy().drop([self.name_target], axis=1).iloc[train_index, :]
            train_y = self.data.copy().loc[train_index, [self.name_target]]
            test_X = self.data.copy().drop([self.name_target], axis=1).iloc[test_index, :]
            test_y = self.data.copy().loc[test_index, [self.name_target]]
            model.fit(train_X, train_y)
            # 测试
            y_pred = model.predict(test_X)
            test_result['RMSE'].append(np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
            test_result['R2'].append(metrics.r2_score(test_y, y_pred))
            test_result['MAE'].append(metrics.mean_absolute_error(test_y, y_pred))

        for key, values in test_result.items():
            test_result[key] = np.array(values).mean()

        return model, test_result, train_X,test_X,test_y


    def GBreg(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion="friedman_mse", 
              min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3, 
              min_impurity_decrease=0., min_impurity_split=1e-7, random_state=None, max_features=None, 
              alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, 
              n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):

        model = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, 
                                          subsample=subsample, criterion=criterion,
                                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth,
                                          min_impurity_decrease=min_impurity_decrease,
                                          min_impurity_split=min_impurity_split, random_state=random_state,
                                          max_features=max_features, alpha=alpha, verbose=verbose,
                                          max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
                                          validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                                          tol=tol, ccp_alpha=ccp_alpha)
        kf = KFold(n_splits=self.cv)
        test_result = {'RMSE': [], 'R2': [], 'MAE': []}
        for train_index, test_index in kf.split(self.data.copy()):
            train_X = self.data.copy().drop(
                [self.name_target], axis=1).iloc[train_index, :]
            train_y = self.data.copy().loc[train_index, [self.name_target]]
            test_X = self.data.copy().drop(
                [self.name_target], axis=1).iloc[test_index, :]
            test_y = self.data.copy().loc[test_index, [self.name_target]]
            model.fit(train_X, train_y)
            # 测试
            y_pred = model.predict(test_X)
            test_result['RMSE'].append(
                np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
            test_result['R2'].append(metrics.r2_score(test_y, y_pred))
            test_result['MAE'].append(
                metrics.mean_absolute_error(test_y, y_pred))

        for key, values in test_result.items():
            test_result[key] = np.array(values).mean()

        return model, test_result, train_X,test_X,test_y

    def Breg(self, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, 
             bootstrap_features=False, warm_start=False, n_jobs=None, random_state=None, verbose=0):

        model = BaggingRegressor(n_estimators=n_estimators, max_samples=max_samples, 
                                 max_features=max_features,bootstrap=bootstrap,
                                 bootstrap_features=bootstrap_features, warm_start=warm_start, n_jobs=n_jobs,
                                 random_state=random_state, verbose=verbose)
        kf = KFold(n_splits=self.cv)
        test_result = {'RMSE': [], 'R2': [], 'MAE': []}
        for train_index, test_index in kf.split(self.data.copy()):
            train_X = self.data.copy().drop(
                [self.name_target], axis=1).iloc[train_index, :]
            train_y = self.data.copy().loc[train_index, [self.name_target]]
            test_X = self.data.copy().drop(
                [self.name_target], axis=1).iloc[test_index, :]
            test_y = self.data.copy().loc[test_index, [self.name_target]]
            model.fit(train_X, train_y)
            # 测试
            y_pred = model.predict(test_X)
            test_result['RMSE'].append(
                np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
            test_result['R2'].append(metrics.r2_score(test_y, y_pred))
            test_result['MAE'].append(
                metrics.mean_absolute_error(test_y, y_pred))

        for key, values in test_result.items():
            test_result[key] = np.array(values).mean()

        return model, test_result, train_X,test_X,test_y

    def XGBreg(self,verbosity=1,eta=0.3,gamma=0,max_depth=6,min_child_weight=1,max_delta_step=0,
               subsample=1,colsample_bytree=1,colsample_bylevel=1,colsample_bynode=1,lambda_=1,alpha=0,
               sketch_eps=0.03,scale_pos_weight=1,refresh_leaf=1,
               max_leaves=0,max_bin=256,num_parallel_tree=1):
    
        model = XGBRegressor(verbosity=verbosity,eta=eta,gamma=gamma,max_depth=max_depth,
                             min_child_weight=min_child_weight,max_delta_step=max_delta_step,subsample=subsample,
                             colsample_bytree=colsample_bytree,colsample_bylevel=colsample_bylevel,
                             colsample_bynode=colsample_bynode,lambda_=lambda_,alpha=alpha,
                             sketch_eps=sketch_eps,scale_pos_weight=scale_pos_weight,
                             refresh_leaf=refresh_leaf,max_leaves=max_leaves,max_bin=max_bin,
                             num_parallel_tree=num_parallel_tree)
        kf = KFold(n_splits=self.cv)
        test_result = {'RMSE' : [], 'R2' : [], 'MAE' : []}
        for train_index, test_index in kf.split(self.data.copy()):
            train_X = self.data.copy().drop([self.name_target], axis=1).iloc[train_index, :]
            train_y = self.data.copy().loc[train_index, [self.name_target]]
            test_X = self.data.copy().drop([self.name_target], axis=1).iloc[test_index, :]
            test_y = self.data.copy().loc[test_index, [self.name_target]]
            model.fit(train_X, train_y)
            # 测试
            y_pred = model.predict(test_X)
            test_result['RMSE'].append(np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
            test_result['R2'].append(metrics.r2_score(test_y, y_pred))
            test_result['MAE'].append(metrics.mean_absolute_error(test_y, y_pred))

        for key, values in test_result.items():
            test_result[key] = np.array(values).mean()

        return model, test_result, train_X,test_X,test_y
