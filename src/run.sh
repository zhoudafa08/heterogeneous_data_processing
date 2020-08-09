#### 1. Matlab
#1.1 Generate time-frequency features of close price, and labels
cd ./ITD
# It is more convenient to operate in the graphical interface environment of Matlab
# run sliding_itd(date,close, 50, 6,1), where date and close denote the trading date and the daily close price.

#### 2. Python
# 2.1 News abstract
python news_abstract_sina_baidu.py 1>../data/300676_sina_baidu_wind_abstract.txt

# 2.2 Convert news abstract to vectors
python snownlp_senta_news.py

# 2.3 Generate Alpha 101 indicators 
python Alpha101.py 

# 2.4 Generate Alpha 191 indicators
python Alpha191.py

# 2.5 Learn and predict (take the classification task as an example)
# 2.5.1 LR, SVM or GBDT model
python traditional_model.py [lr/svm/gbdt]

# 2.5.2 LSTM model
python lstm.py

# 2.5.3 LSTM_Attention model
python lstm_attention.py
