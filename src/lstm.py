#__*__encoding=utf-8__*__
#Written by Feng Zhou(fengzhou@gdufe.edu.cn)

import os, sys, time, datetime
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import Input, Model, losses, Sequential
from keras.activations import relu, sigmoid
from keras.layers import Dense, Bidirectional, LSTM, Dropout, Activation, Flatten, concatenate, Reshape
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
import warnings
import random
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
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


def build_model(x_train, y_train, x_validation, y_validation, x_test, y_test):
    print(x_train.shape, y_train.shape)
    print(x_validation.shape, y_validation.shape)
    print(x_test.shape, y_test.shape)
    model = Sequential()
    model.add(LSTM({{choice([16, 32, 64, 128])}} , input_shape=x_train.shape[1:], return_sequences=False))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([64, 128, 256])}}))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([8, 16, 32])}}))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss=losses.binary_crossentropy, optimizer=Adam(1e-4), metrics=['accuracy'])
    filepath = './models/'+'lstm_model.h5'
    #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    result = model.fit(x_train, y_train, 
                batch_size=32, 
		epochs={{choice(range(10, 150, 20))}}, 
		#verbose=1,
		#callbacks=[checkpoint],
		#validation_split=0.1
		validation_data=(x_validation, y_validation))
    #validation_acc = np.amax(result.history['val_accuracy'])
    score, acc = model.evaluate(x_validation, y_validation, verbose=0)
    try:
        with open('metric.txt') as f:
            min_acc = float(f.read().strip())
    except FileNotFoundError:
        min_acc = -acc
    if -acc <= min_acc:
        model.save(filepath)
        with open('metric.txt', 'w') as f:
            f.write(str(-acc))
    sys.stdout.flush()
    return {'loss': -acc, 'model': model, 'status': STATUS_OK}

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
    data_freq_spct = pd.DataFrame(np.c_[dates,IAs,IPs,imfs],columns=['date','ia1','ia2','ia3','ia4','ia5','ip1', 'ip2', 'ip3', 'ip4','ip5','imf1','imf2','imf3','imf4','imf5','imf6'])
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

    # (2.5) feature dimension reduction
    pca = PCA(n_components=5)
    pca_results = pca.fit_transform(new_alpha191)
    print(np.sum(pca.explained_variance_ratio_))
    pca_results = pd.DataFrame(pca_results, columns=['alpha191_vect1', 'alpha191_vect2', 'alpha191_vect3', 'alpha191_vect4', 'alpha191_vect5'], index=fin_index) 
    
    tsne = PCA(n_components=5)
    tsne_results = tsne.fit_transform(new_alpha101)
    print(np.sum(tsne.explained_variance_ratio_))
    tsne_results = pd.DataFrame(tsne_results, columns=['alpha101_vect1', 'alpha101_vect2', 'alpha101_vect3', 'alpha101_vect4', 'alpha101_vect5'], index=fin_index)
    
    # (2.6) Concat EMD & frequency spectrum, Alpha101, Alpha191 
    sample_feats = np.array(pd.concat([trade_data, new_data_freq_spct, pca_results, tsne_results, news_rate], axis=1))
    #print('labels:', data_labels.shape)
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
    feats = np.zeros((sample_feats.shape[0]-time_steps+1, time_steps, sample_feats.shape[1]))
    for i in range(feats.shape[0]):
        feats[i, :, :] = sample_feats[i:i+time_steps, :]
        labels[i] = data_labels.iloc[i+time_steps-1] 
    train_num = int(len(labels) * 0.7)
    validation_num = int(len(labels) * 0.2)
    y_train = labels[:train_num]
    y_validation = labels[train_num:train_num+validation_num]
    y_test = labels[train_num+validation_num:]
    x_train = feats[:train_num]
    x_validation = feats[train_num:train_num+validation_num]
    x_test = feats[train_num+validation_num:]
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return x_train, y_train, x_validation, y_validation, x_test, y_test

if __name__ =='__main__':
    best_run, best_model = optim.minimize(model=build_model, data=data, algo=tpe.suggest, max_evals=20, trials=Trials())
    x_train, y_train, x_validation, y_validation, x_test, y_test = data()
    best_model = load_model('./models/lstm_model.h5')
    # train
    preds_prob = best_model.predict(x_train)
    preds=preds_prob.copy()
    cond=preds>=0.5
    preds[cond] = 1
    preds[~cond]=0
    true_labels=np.array(y_train, dtype=int)
    train_prec, train_recall, train_f1 = prec_recall_f1(true_labels, preds)
    train_acc = accuracy_score(preds, true_labels)
    train_auc = roc_auc_score(true_labels, preds_prob)
    print('Train accuracy:', train_acc)
    print('Train precision:', train_prec)
    print('Train recall:', train_recall)
    print('Train f1:', train_f1)
    print('Train AUC:', train_auc)
    # validation
    preds_prob = best_model.predict(x_validation)
    preds=preds_prob.copy()
    cond=preds>=0.5
    preds[cond] = 1
    preds[~cond]=0
    true_labels=np.array(y_validation, dtype=int)
    train_prec, train_recall, train_f1 = prec_recall_f1(true_labels, preds)
    train_acc = accuracy_score(preds, true_labels)
    train_auc = roc_auc_score(true_labels, preds_prob)
    print('validation accuracy:', train_acc)
    print('validation precision:', train_prec)
    print('validation recall:', train_recall)
    print('validation f1:', train_f1)
    print('validation AUC:', train_auc)
    # test
    preds_prob = best_model.predict(x_test)
    preds=preds_prob.copy()
    cond=preds>=0.5
    preds[cond] = 1
    preds[~cond]=0
    true_labels=np.array(y_test, dtype=int)
    test_prec, test_recall, test_f1 = prec_recall_f1(true_labels, preds)
    test_acc = accuracy_score(preds, true_labels)
    test_auc = roc_auc_score(true_labels, preds_prob)
    print('Test accuracy:', test_acc)
    print('Test precision:', test_prec)
    print('Test recall:', test_recall)
    print('Test f1:', test_f1)
    print('Test AUC:', test_auc)
    
    print("LSTM trained and tested successful!")
    
