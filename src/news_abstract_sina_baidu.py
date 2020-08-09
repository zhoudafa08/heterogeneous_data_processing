#_*_coding:utf-8_*_
#Written by Feng Zhou(fengzhou@gdufe.edu.cn)
#Run command: python news_abstract_sina_baidu.py 1>../data/300676_sina_baidu_wind_abstract.txt

from textrank4zh import TextRank4Sentence
import sys
import numpy as np
import pickle
import datetime
import pandas as pd
import time, datetime

start='2017-07-14'
end='2020-07-21'
date_start=datetime.datetime.strptime(start, '%Y-%m-%d')
date_end=datetime.datetime.strptime(end, '%Y-%m-%d')

date_wnd = []
while date_start < date_end:
    date_start += datetime.timedelta(days=1)
    date_wnd.append(date_start.strftime('%Y-%m-%d'))

file_path1='../data/300676_sina.txt' #sina news
file_lines=open(file_path1, 'r')
file_path2='../data/300676_baidu.txt' #baidu news
file_lines2=open(file_path2, 'r')
file_path3='../data/300676_news_wind.xls' # wind news
file_lines3 = pd.read_excel(file_path3)

date_news={}
date_news_abstract={}

for line in file_lines.readlines(): #read sina news
    line=line.strip(' ').split('\t')
    date=line[0][:10]
    if len(line) != 3:
        continue
    if date in date_news.keys():
        date_news[date] += line[2] 
    else:
        date_news[date] = line[2]
file_lines.close()

for line in file_lines2.readlines(): #read baidu news
    line=line.strip(' ').split('\t')
    if len(line[0])<11:
        continue
    date=line[0][:4]+'-'+line[0][7:9]+'-'+line[0][12:14]
    if len(line) != 5:
        continue
    if date in date_news.keys():
        date_news[date] += line[4] 
    else:
        date_news[date] = line[4]
file_lines2.close()

for index, row in file_lines3.iterrows():
    date = str(row['date'])[:10]
    if date in date_news.keys():
        date_news[date] += (row['title'] + '。')
        date_news[date] += row['news'].replace('\n', '') 
    else:
        date_news[date] = (row['title'] + '。')
        date_news[date] = row['news'].replace('\n', '') 

tr4s=TextRank4Sentence()
for date in date_wnd:
    if date in date_news:
        tr4s.analyze(date_news[date], lower=True, source='all_filters')
        abstract=[]
        for item in tr4s.get_key_sentences(num=100):
            if len(item.sentence) < 100:
                abstract.append([item.index, item.sentence])
        abstract=sorted(abstract[:10], key=lambda x: x[0])
        abstract=["%s。" % (x[1]) for i, x in enumerate(abstract, 1)]
        print(date, '\t', "".join(abstract))
    else:
        print(date, '\t')
        
