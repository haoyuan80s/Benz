import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def main():
    favorite_color = { "lion": "yellow", "kitty": "red" }
    pickle.dump( favorite_color, open( "save.p", "wb" ) )

    pickle.load( open( "save.p", "rb" ) )

    # read datasets
    train = pd.read_csv('data_fun/data/train.csv')
    test = pd.read_csv('data_fun/data/test.csv')

    # process columns, apply LabelEncoder to categorical features
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values) + list(test[c].values))
            train[c] = lbl.transform(list(train[c].values))
            test[c] = lbl.transform(list(test[c].values))

    # shape
    print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

    # This step is huge!
    from sklearn.decomposition import PCA, FastICA
    n_comp = 10

    # PCA
    pca = PCA(n_components=n_comp, random_state=42)
    pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
    pca2_results_test = pca.transform(test)

    # ICA
    ica = FastICA(n_components=n_comp, random_state=42)
    ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
    ica2_results_test = ica.transform(test)

    # Append decomposition components to datasets
    for i in range(1, n_comp+1):
        train['pca_' + str(i)] = pca2_results_train[:,i-1]
        test['pca_' + str(i)] = pca2_results_test[:, i-1]
        train['ica_' + str(i)] = ica2_results_train[:,i-1]
        test['ica_' + str(i)] = ica2_results_test[:, i-1]

    y_train = train["y"]
    y_mean = np.mean(y_train)

    y_train = train["y"]
    y_mean = np.mean(y_train)
    ()# mmm, xgboost, loved by everyone ^-^
    import xgboost as xgb
    xgb_params = {
        'n_estimators': 550,
        'learning_rate': 0.005,
        'max_depth': 4,
        'subsample': 0.95,
        'objective': 'reg:linear',
        #'eval_metric': 'rmse',
        'base_score': y_mean, # base prediction = mean(target)
        #'silent': 1
    }

    boost = xgb.XGBRegressor(**xgb_params)

    parameters = {
        # n_estimators
        'learning_rate': [0.01, 0.015,0.05],
        'gamma': [0,0.1,0.5,0.9],
        'max_depth': [4, 9],
        'min_child_weight': [1,5],
        "subsample": [0.6,0.8,1],
        'colsample_bytree': [0.6,0.8,1],
        'reg_alpha' : [0],
        'reg_lambda' : [1],
    }
    reg = GridSearchCV(boost, parameters, n_jobs=1, cv=3, verbose = 2)
    reg.fit(train.drop('y', axis=1).as_matrix(), y_train)

    best_parameters, score, _ = max(reg.grid_scores_, key=lambda x: x[1])
    print(score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    pickle.dump( reg.best_params_, open("bestpara.p", "wb" ))


    dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
    dtest = xgb.DMatrix(test)


    def r2_metric(preds, dtrain):
        """Self defined evaluation obj"""
        from sklearn.metrics import r2_score
        return 'r2_metric', r2_score(dtrain.get_label(), preds)
                
    # xgboost, cross-validation
    cv_result = xgb.cv(reg.best_params_,
                       dtrain,
                       num_boost_round=1000, # increase to have better results (~700)
                       early_stopping_rounds=50,
                       verbose_eval=50,
                       show_stdv=False,
                       feval = r2_metric,
                       maximize = True
    )

    num_boost_rounds = len(cv_result)
    print(num_boost_rounds)

    # train model
    model = xgb.train(dict(reg.best_params_, silent=0), dtrain, num_boost_round=num_boost_rounds)


    # check f2-score (to get higher score - increase num_boost_round in previous cell)

    # now fixed, correct calculation
    print(r2_score(dtrain.get_label(), model.predict(dtrain)))

    # make predictions and save results
    y_pred = model.predict(dtest)
    output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
    output.to_csv('xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)

if __name__ == '__main__':
    main()
