#_*_coding:utf-8_*_
#Written by Feng Zhou(fengzhou@gdufe.edu.cn)
#Run command: python snownlp_senta_news.py

from __future__ import division
import codecs
import numpy as np
from snownlp import SnowNLP
import pandas as pd
import pickle
import paddlehub as hub

if __name__ =='__main__':
    file_path='../data/300676_sina_baidu_wind_abstract.txt'
    file_lines=codecs.open(file_path, 'r', 'utf-8')
    date = []
    rate = np.empty((1,4), dtype=float)
    senta = hub.Module(name='senta_bilstm')
    for line in file_lines.readlines():
        line = line.strip(' ').split('\t')
        date.append(line[0])
        temp_rate = []
        temp_senta = []
        if len(line[1].strip())<10:
            rate = np.r_[rate, np.array([[0.5, 0.0, 0.5, 0.0]])]
        else:
            sents=line[1].strip(' ').split('。')
            for sent in sents:
                fenshu = SnowNLP(sent).sentiments
                temp_rate.append(fenshu)
                input_dict={'text':[sent]}
                results = senta.sentiment_classify(data=input_dict)
                temp_senta.append(results[0]['positive_probs'])
            rate = np.r_[rate, np.array([[np.mean(temp_rate), np.std(temp_rate), np.mean(temp_senta), np.std(temp_senta)]])]
    file_lines.close()
    rate = np.delete(rate, 0, axis=0)
    print(rate, rate.shape)
    np.save('../data/300676_rate_news.npy', rate)
    np.save('../data/300676_date_news.npy', date)
