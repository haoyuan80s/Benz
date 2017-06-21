import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

def data_with_val():
    print('generating data......')
    # read datasets
    train_total = pd.read_csv('../data/train.csv') ## Shape train: (4209, 378)
    X_test = pd.read_csv('../data/test.csv') ## Shape test: (4209, 377)

    # Shuffle data
    np.random.seed(0)
    l = [x for x in range(4209)]
    np.random.shuffle(l)
    train_total = train_total.iloc[l]

    # split data
    ratio = 0.6
    threshold = int(ratio*4209)
    X_train = train_total.iloc[range(threshold)] 
    val = train_total.iloc[range(threshold, 4209)]
    y_train = X_train['y']
    X_train = X_train.drop('y', axis = 1)
    y_val = val['y']
    X_val = val.drop('y', axis = 1)

    # process type
    for c in train_total.columns:
        if train_total[c].dtype == 'object':
            lbl = LabelEncoder() 
            lbl.fit(list(train_total[c].values) + list(X_test[c].values)) 
            X_train[c] = lbl.transform(list(X_train[c].values))
            X_val[c] = lbl.transform(list(X_val[c].values))
            X_test[c] = lbl.transform(list(X_test[c].values))

    # shape        
    print('Shape X_train:', X_train.shape)
    print('Shape X_test:', X_test.shape)
    print('Shape X_val:', X_val.shape )
    return X_train, y_train, X_val, y_val, X_test

def data():
    print('generating data......')
    # read datasets
    X_train = pd.read_csv('../data/train.csv') ## Shape train: (4209, 378)
    X_test = pd.read_csv('../data/test.csv') ## Shape test: (4209, 377)

#     # Shuffle data
#     l = [x for x in range(4209)]
#     np.random.seed(0)
#     np.random.shuffle(l)
#     X_train = X_train.iloc[l]

    y_train = X_train['y']
    X_train = X_train.drop('y', axis = 1)

    # process type
    for c in X_train.columns:
        if X_train[c].dtype == 'object':
            lbl = LabelEncoder() 
            lbl.fit(list(X_train[c].values) + list(X_test[c].values)) 
            X_train[c] = lbl.transform(list(X_train[c].values))
            X_test[c] = lbl.transform(list(X_test[c].values))

    # shape        
    print('Shape X_train:', X_train.shape)
    print('Shape X_test:', X_test.shape)
    return X_train, y_train, X_test



class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        params = params.copy() # creat a local version of params
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))
    

def get_oof(clf, x_train, y_train, x_test):
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def turing_xgb(N):
    best_err = 100000000
    data = []
    for _ in range(N):
        params = {
            'colsample_bytree': np.random.uniform(0.01,1),
            'subsample': np.random.uniform(0.01,1),
            'learning_rate': np.exp(np.random.uniform(np.log(0.001),np.log(0.1))),
            'objective': 'reg:linear',
            'max_depth': int(np.random.uniform(1,10)),
            'num_parallel_tree': int(np.random.uniform(1,3)),
            'min_child_weight': int(np.random.uniform(1,5)),
            'nrounds': int(np.random.uniform(300,800))
        }
        model = XgbWrapper(seed=0, params=params)
        oof_train, oof_test = get_oof(model)
        err = mean_squared_error(y_train, oof_train)
        if best_err > err:
            best_err = err
            best_para = params
        print(err,best_err, params)
        data += [(params,err, oof_train, oof_test)]

    pickle.dump(data,open("xgb.p",'wb'))    
    
def turing(clf,N,name):
    best_err = 100000000
    data = []
    for _ in range(N):
        n_estimators = int(np.random.uniform(1,900))
        max_features = np.random.uniform(0.01,1)
        max_depth = int(np.random.uniform(1,10))
        min_samples_leaf = int(np.random.uniform(1,7))
        params = {
            'n_jobs': 8,
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
        }
        model = SklearnWrapper(clf=clf, seed=0, params=params)
        oof_train, oof_test = get_oof(model)
        err = mean_squared_error(y_train, oof_train)
        if best_err > err:
            best_err = err
            best_para = params
        print(err,best_err, params)
        data += [(params,err,oof_train, oof_test)]

    pickle.dump(data,open("{}.p".format(name),'wb'))    
