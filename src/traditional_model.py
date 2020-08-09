#__*__encoding=utf-8__*__
#Written by Feng Zhou(fengzhou@gdufe.edu.cn)

import os, sys, time, datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import warnings
import random
from hyperopt import fmin, hp, Trials, STATUS_OK, tpe
from sklearn.externals import joblib
from matplotlib import pyplot

warnings.filterwarnings('ignore')
def prec_recall_f1(true_labels, preds):
    preds = preds.reshape(true_labels.shape)
    temp=true_labels.copy()
    cond = (true_labels==1) & (preds==1)
    temp[cond] = 1
    temp[~cond] = 0
    TP= np.sum(temp)
    cond = (true_labels==1) & (preds==0)
    temp[cond] = 1
    temp[~cond] = 0
    FP=np.sum(temp)
    cond = (true_labels==0) & (preds==1)
    temp[cond] = 1
    temp[~cond] = 0
    FN=np.sum(temp)
    cond = (true_labels==0) & (preds==0)
    temp[cond] = 1
    temp[~cond] = 0
    TN=np.sum(temp)
    prec = TP / (TP+FP)
    recall = TP / (TP+FN)
    return prec, recall, 2*prec*recall/(prec+recall)

def data():
    # (1)Load Data
    # (1.1) Trade data and Alpha101 & Alpha191 indices
    path = '../data/alpha101_0721_300676.xls'
    alpha101 = pd.read_excel(path)
    #alpha101['date'] = alpha101['date'].apply(lambda x:x.strftime('%Y-%m-%d')) 
    alpha101.set_index(['date'], inplace=True)
    path = '../data/alpha191_0721_300676.xls' 
    alpha191 = pd.read_excel(path)
    #alpha191['date'] = alpha191['date'].apply(lambda x:x.strftime('%Y-%m-%d')) 
    alpha191.set_index(['date'], inplace=True)
    
    # (1.2) EMD & frequency spectrum features & labels
    dates = pd.read_csv('../data/huada_itd_date_0721.csv')
    imfs = pd.read_csv('../data/huada_PRC_0721.csv', header=None)
    IAs = pd.read_csv('../data/huada_IA_0721.csv', header=None)
    IPs = pd.read_csv('../data/huada_IP_0721.csv', header=None)
    data_freq_spct = pd.DataFrame(np.c_[dates,IAs,IPs,imfs],columns=['date','a1','a2','a3','a4','a5','p1', 'p2', 'p3', 'p4','p5','c1','c2','c3','c4','c5','c6'])
    data_freq_spct.set_index(['date'], inplace=True)
    
    # (1.3) Labels
    labels = pd.read_csv('../data/huada_label_0721.csv', header=None)
    dates = pd.read_csv('../data/huada_itd_date_0721.csv')
    data_labels = pd.DataFrame(np.c_[dates, labels],columns=['date','label'])
    data_labels.set_index(['date'], inplace=True)

    # (2) Data Cleaning & alignment & normalization
    # (2.1) Alpha101 & Alpha191 & trade data
    abnormal_ratio_101 = ((alpha101 == np.inf).sum() + (alpha101 == -np.inf).sum() + alpha101.isnull().sum()) / len(alpha101) 
    abnormal_ratio_191 = ((alpha191 == np.inf).sum() + (alpha191 == -np.inf).sum() + alpha191.isnull().sum()) / len(alpha191)
    temp = (abnormal_ratio_101 > 0.03)
    for indices in temp[temp].index.tolist():
        del alpha101[indices]
    del alpha101['alpha1_084']
    temp = (abnormal_ratio_191 > 0.03)
    for indices in temp[temp].index.tolist():
        del alpha191[indices]
    del alpha191['alpha017']
    del alpha191['open']
    del alpha191['high']
    del alpha191['low']
    del alpha191['close']
    del alpha191['returns']
    del alpha191['volume']
    del alpha191['vwap']
    del alpha191['cap']
    del alpha191['prev_close']
    del alpha191['amount']
    
    new_alpha101 = alpha101.copy() 
    for indices in alpha101.columns.tolist():
        if alpha101[indices].dtypes != bool:
            where_inf = np.isinf(alpha101[indices])
            alpha101[indices][where_inf] = np.nan
            cond = alpha101[indices].isna()
            new_alpha101[indices][~cond] = preprocessing.scale(alpha101[indices].dropna())
            new_alpha101[indices][cond] = 0
        else:
            temp_data = pd.get_dummies(alpha101[indices])
            temp_data.columns=[indices+'_False',indices+'_True']
            del new_alpha101[indices]
            new_alpha101 = new_alpha101.join(temp_data)

    fin_index = data_labels.index.tolist()
    new_alpha101 = pd.DataFrame(new_alpha101, index=fin_index)
    new_alpha101.index.set_names(['date'],inplace=True)
    trade_data = new_alpha101.iloc[:,0:11]
    trade_data = pd.DataFrame(trade_data, index=fin_index)
    new_alpha101 = new_alpha101.iloc[:,11:]
    new_alpha101 = pd.DataFrame(new_alpha101, index=fin_index)

    new_alpha191 = alpha191.copy() 
    for indices in alpha191.columns.tolist():
        if alpha191[indices].dtypes != bool:
            where_inf = np.isinf(alpha191[indices])
            alpha191[indices][where_inf] = np.nan
            cond = alpha191[indices].isna()
            new_alpha191[indices][~cond] = preprocessing.scale(alpha191[indices].dropna())
            new_alpha191[indices][cond] = 0
        else:
            temp_data = pd.get_dummies(alpha191[indices])
            temp_data.columns=[indices+'_False',indices+'_True']
            del new_alpha191[indices]
            new_alpha191 = new_alpha191.join(temp_data)

    fin_index = data_labels.index.tolist()
    new_alpha191 = pd.DataFrame(new_alpha191, index=fin_index)
    new_alpha191.index.set_names(['date'],inplace=True)

    # (2.2) EMD & frequency spectrum   
    new_data_freq_spct = data_freq_spct.copy()
    for indices in data_freq_spct.columns.tolist():
        new_data_freq_spct[indices] = preprocessing.scale(data_freq_spct[indices])  
    
    # (2.3) News sentiments
    date_news = np.load('../data/300676_date_news.npy')
    rate_news = np.load('../data/300676_rate_news.npy')
    snow_avg_dict = {}
    snow_std_dict = {}
    senta_avg_dict = {}
    senta_std_dict = {}
    for i in range(date_news.shape[0]):
        date = np.datetime64(str(date_news[i].strip()))
        snow_avg_dict[np.datetime_as_string(date)] = rate_news[i, 0]
        snow_std_dict[np.datetime_as_string(date)] = rate_news[i, 1]
        senta_avg_dict[np.datetime_as_string(date)] = rate_news[i, 2]
        senta_std_dict[np.datetime_as_string(date)] = rate_news[i, 3]
    news_rate = np.zeros((len(data_labels), 4))
    i = 0
    for date in data_labels.index.tolist():
        if date in snow_avg_dict:
            news_rate[i, 0] = snow_avg_dict[date]
            news_rate[i, 1] = snow_std_dict[date]
            news_rate[i, 2] = senta_avg_dict[date]
            news_rate[i, 3] = senta_std_dict[date]
            i += 1
        else:
            continue
    news_rate = pd.DataFrame(np.c_[np.array(data_labels.index), news_rate],columns=['date', 'snow_avg','snow_std', 'senta_avg', 'senta_std'])
    news_rate.set_index(['date'], inplace=True)
    
    # (2.4) pandas data describe
    alpha101 = pd.DataFrame(alpha101, index=fin_index)
    alpha101.index.set_names(['date'],inplace=True)
    alpha191 = pd.DataFrame(alpha191, index=fin_index)
    alpha191.index.set_names(['date'],inplace=True)
    old_trade_data = alpha101.iloc[:,0:11]
    pandas_data1 = pd.concat([old_trade_data], axis=1)
    print(pandas_data1)
    pandas_data2 = pd.concat([alpha101.iloc[:, 11:]])
    pandas_data3 = pd.concat([alpha191])
    pandas_data4 = pd.concat([data_freq_spct, news_rate], axis=1)
    pandas_data4 = pandas_data4.astype(float)
    print(pandas_data1.describe())
    print(pandas_data2.describe())
    print(pandas_data3.describe())
    print(pandas_data4.describe())
    #pandas_data1.describe().to_excel('df1.xls')
    #pandas_data2.describe().to_excel('df2.xls')
    #pandas_data3.describe().to_excel('df3.xls')
    #pandas_data4.describe().to_excel('df4.xls')

    # (2.5) feature dimension reduction
    pca = PCA(n_components=5)
    pca_results = pca.fit_transform(new_alpha191)
    print(np.sum(pca.explained_variance_ratio_))
    pca_results = pd.DataFrame(pca_results, columns=['alpha191_v1', 'alpha191_v2', 'alpha191_v3', 'alpha191_v4', 'alpha191_v5'], index=fin_index) 
    
    tsne = PCA(n_components=5)
    tsne_results = tsne.fit_transform(new_alpha101)
    print(np.sum(tsne.explained_variance_ratio_))
    tsne_results = pd.DataFrame(tsne_results, columns=['alpha101_v1', 'alpha101_v2', 'alpha101_v3', 'alpha101_v4', 'alpha101_v5'], index=fin_index)
    
    # (2.6) Concat EMD & frequency spectrum, Alpha101, Alpha191 
    pd_sample_feats = pd.concat([trade_data, new_data_freq_spct, pca_results, tsne_results, news_rate], axis=1)
    sample_feats = np.array(pd_sample_feats)
    #print('labels:', data_labels.shape)
    #print('news vectors:', news_vects.shape)
    #print('trade data:', trade_data.shape)
    #print('news rate:', news_rate.shape)
    #print('alpha 101:', tsne_results.shape) 
    #print('alpha 191:', pca_results.shape) 
    #print('EMD and frequency spectrum:', new_data_freq_spct.shape) 
    #print('Concated feats:', sample_feats.shape)
    
    # (3) Model Design & Train, Test 
    # (3.1) Data Division
    time_steps = 5
    labels = np.zeros((sample_feats.shape[0]-time_steps+1))
    feats = np.zeros((sample_feats.shape[0]-time_steps+1, time_steps*sample_feats.shape[1]))
    for i in range(feats.shape[0]):
        feats[i, :] = sample_feats[i:i+time_steps, :].reshape(1, time_steps*sample_feats.shape[1])
        labels[i] = data_labels.iloc[i+time_steps-1] 
    train_num = int(len(labels) * 0.7)
    validation_num = int(len(labels) * 0.2)
    y_train = labels[:train_num]
    y_validation = labels[train_num:train_num+validation_num]
    y_test = labels[train_num+validation_num:]
    x_train = feats[:train_num]
    x_validation = feats[train_num:train_num+validation_num]
    x_test = feats[train_num+validation_num:]
    return x_train, y_train, x_validation, y_validation, x_test, y_test, pd_sample_feats.columns.tolist()

def build_model(paras):
    x_train = paras['x_train']
    y_train = paras['y_train']
    x_validation = paras['x_validation']
    y_validation = paras['y_validation']
    model_name = paras['model_name']
    del paras['x_train']
    del paras['x_validation']
    del paras['y_train']
    del paras['y_validation']
    del paras['x_test']
    del paras['y_test']
    del paras['model_name']
    if model_name == 'lr':
        model = LogisticRegression(**paras)
    elif model_name == 'svm':
        model = SVC(**paras, probability=True)
    elif model_name == 'gbdt':
        model = xgb.XGBClassifier(**paras)
    model.fit(x_train, y_train)
    preds = model.predict(x_validation)
    y_true = y_validation
    y_pred = preds
    acc = accuracy_score(np.array(y_true), y_pred)
    try:
        with open('metric.txt') as f:
            min_acc = float(f.read().strip())
    except FileNotFoundError:
        min_acc = -acc
    if -acc <= min_acc:
        print(acc)
        joblib.dump(model, './models/'+model_name+'_feat_processing.model')
        with open('metric.txt', 'w') as f:
            f.write(str(-acc))
    sys.stdout.flush()
    return {'loss': -acc, 'status': STATUS_OK}

def train_test(y_train, x_train, y_validatioin, x_validation, y_test, x_test, model_name):
    if model_name == 'lr':
        paras = {
	   'x_train': x_train,
	   'x_test': x_test,
	   'x_validation': x_validation,
	   'y_train': y_train,
	   'y_test': y_test,
	   'y_validation': y_validation,
	   'C': hp.choice('C', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 0.005, 0.01, 0.05, 0.1, 1, 5, 10]),
	   #'C': hp.choice('C', range(1, 250)),
           'penalty': hp.choice('penalty', ['l2']),
	   'max_iter': hp.choice('max_iter', range(100, 500)),
	   'model_name': hp.choice('model_name', ['lr']),
           'solver': hp.choice('solver', ['lbfgs', 'sag', 'saga', 'newton-cg', 'liblinear'])
        }
    elif model_name == 'svm':
        paras = {
	   'x_train': x_train,
	   'x_validation': x_validation,
	   'x_test': x_test,
	   'y_train': y_train,
	   'y_validation': y_validation,
	   'y_test': y_test,
          'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid', 'linear']),
          'C': hp.choice('C', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]),
	   'max_iter': hp.choice('max_iter', range(50, 250, 50)),
	  'gamma': hp.choice('gamma', ['scale', 'auto']),
	   'model_name': hp.choice('model_name', ['svm']),
          'degree': hp.choice('degree', [1, 2, 3, 4, 5])
        }
    elif model_name == 'gbdt':
        paras = {
	   'x_train': x_train,
	   'x_validation': x_validation,
	   'x_test': x_test,
	   'y_train': y_train,
	   'y_validation': y_validation,
	   'y_test': y_test,
          'n_estimators': hp.choice('n_estimators', range(10, 250)),
          'learning_rate': hp.choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
	   'max_iter': hp.choice('max_iter', range(200, 800, 10)),
	   'max_depth': hp.choice('max_depth', range(1, 8)),
	   'gamma': hp.choice('gamma', [0.01, 0.1, 0.5, 1]),
	 # 'alpha': hp.choice('alpha', range(1, 100, 2)),
	 #  'lambda': hp.choice('lambda', range(1, 100, 2)),
	   'model_name': hp.choice('model_name', ['gbdt'])
        }
    best = fmin(build_model, paras, algo=tpe.suggest, max_evals=200, trials=Trials())
    print('Best:', best)
    return 0

if __name__ =='__main__':
    x_train, y_train, x_validation, y_validation, x_test, y_test, columns = data()
    pred_results = np.zeros((y_test.shape[0], 3))
    true_labels = y_test
    pred_results[:, 0] = np.array(true_labels).reshape(y_test.shape[0], )
    temp_results = np.zeros((y_test.shape[0],1))
        
    if (os.path.exists('./metric.txt')):
        os.remove('./metric.txt')

    # (3.3) Model Design & Train
    result = train_test(y_train, x_train, y_validation, x_validation, y_test, x_test, sys.argv[1])
    
    # train
    best = joblib.load('./models/'+sys.argv[1]+'_feat_processing.model')
    
    preds = best.predict(x_train)
    print(preds.shape, y_train.shape)
    prec, recall, f1 = prec_recall_f1(y_train, preds)
    print('Train accuracy:', accuracy_score(preds, y_train))
    print('Train precision:', prec)
    print('Train recall:', recall)
    print('Train f1:', f1)
    preds_prob = best.predict_proba(x_train)
    print('Train AUC:', roc_auc_score(y_train, preds_prob[:,1]))
    
    preds = best.predict(x_validation)
    print(preds.shape, y_validation.shape)
    prec, recall, f1 = prec_recall_f1(y_validation, preds)
    print('Train accuracy:', accuracy_score(preds, y_validation))
    print('Train precision:', prec)
    print('Train recall:', recall)
    print('Train f1:', f1)
    preds_prob = best.predict_proba(x_validation)
    print('Train AUC:', roc_auc_score(y_validation, preds_prob[:,1]))
    
    # test
    preds = best.predict(x_test)
    print(true_labels.shape, preds.shape)
    prec, recall, f1 = prec_recall_f1(y_test, preds)
    print('Test accuracy:', accuracy_score(preds, y_test))
    print('Test precision:', prec)
    print('Test recall:', recall)
    print('Test f1:', f1)
    preds_prob = best.predict_proba(x_test)
    print('Test AUC:', roc_auc_score(y_test, preds_prob[:,1]))
