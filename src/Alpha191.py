#_*_coding:utf-8_*_
#Written by Feng Zhou(fengzhou@gdufe.edu.cn)

from __future__ import division
from scipy.stats import rankdata
import scipy as sp
import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from functools import partial

# region Auxiliary functions
def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    
    return df.rolling(window).sum()

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)

def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)

def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)

def ts_count(df, window=10):
    return df.rolling(window).count()

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()

def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    #return df.rank(axis=1, pct=True)
    return df.rank(pct=True)

def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :] 
    na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])  

def func_rank(na):
    return rankdata(na)[-1]/rankdata(na).max()

def func_decaylinear(na):
    n = len(na)
    decay_weights = np.arange(1,n+1,1) 
    decay_weights = decay_weights / decay_weights.sum()
    return (na * decay_weights).sum()

def func_decay_linear(df, window=10):
    return df.rolling(window).apply(func_decaylinear)

def func_highday(na):
    return len(na) - np.argmax(na)

def func_high_day(df,window=10):
    return df.rolling(window).apply(func_highday)

def func_lowday(na):
    return len(na) - np.argmin(na)

def func_low_day(df,window=10):
    return df.rolling(window).apply(func_lowday)

def wma(df):
    weight = np.arange(1, len(df) + 1) / len(df)
    df = df * weight
    return df.sum()

def func_wma(df, window=10):
    return df.rolling(window).apply(wma)

def regbert_slope(x):
    y = pd.Series(range(1, len(x) + 1))
    Model = sp.stats.linregress(x, y)
    return Model.slope

def func_regbert_slope(x, window=10):
    return x.rolling(window).apply(regbert_slope)

def regbert_intercept(x):
    y = pd.Series(range(1, len(x) + 1))
    Model = sp.stats.linregress(x, y)
    return Model.intercept

def func_regbert_intercept(x, window=10):
    return x.rolling(window).apply(regbert_intercept)
# endregion


class GTJA_191(object):
    def __init__(self, df, df_benchmark):
    #def __init__(self, end_date):
        #price = get_price('300676.XSHE', None, end_date, '1d', ['open','close','low','high','avg','pre_close','volume','price'], False, None, 250, panel=True)
        #benchmark_price = get_price('000300.XSHG', None, end_date, '1d',['open','close','low','high','avg','pre_close','volume'], False, None, 250, panel=True)
        #self.open_price = price.loc['open',:,:].dropna(axis=1,how='any')
        #self.close      = price.loc['close',:,:].dropna(axis=1,how='any')
        #self.low        = price.loc['low',:,:].dropna(axis=1,how='any')
        #self.high       = price.loc['high',:,:].dropna(axis=1,how='any')
        #self.avg_price  = price.loc['avg_price',:,:].dropna(axis=1,how='any')
        #self.prev_close = price.loc['prev_close',:,:].dropna(axis=1,how='any')
        #self.volume     = price.loc['volume',:,:].dropna(axis=1,how='any')
        #self.amount     = price.loc['turnover',:,:].dropna(axis=1,how='any')
        #self.benchmark_open_price = benchmark_price.loc[:, 'open']
        #self.benchmark_close_price = benchmark_price.loc[:, 'close']
        self.open_price = df['open']
        self.open = df['open']
        self.close      = df['close']
        self.low        = df['low']
        self.high       = df['high']
        self.avg_price  = df['vwap']
        self.vwap  = df['vwap']
        self.prev_close = df['prev_close']
        self.volume     = df['volume']
        self.amount     = df['amount']
        self.returns     = df['returns']
        self.benchmark_open = df_benchmark['open']
        self.benchmark_close = df_benchmark['close']
        #########################################################################
    
    
    #############################################################################
    
    def alpha_001(self):
        #data1 = self.volume.diff(periods=1).rank(axis=1,pct=True)
        #data2 = ((self.close - self.open_price)/self.open_price).rank(axis=1,pct=True)
        #alpha = -data1.iloc[-6:,:].corrwith(data2.iloc[-6:,:]).dropna()
        #alpha = -data1.iloc[-6:,:].corrwith(data2.iloc[-6:,:]).dropna()
        #alpha=alpha.dropna()
	#return alpha
        return (-1 * correlation(rank(delta(log(self.volume), 1)), rank(((self.close - self.open) / self.open)), 6))
    
    def alpha_002(self):
        ##### -1 * delta((((close-low)-(high-close))/((high-low)),1))####
        #result=((self.close-self.low)-(self.high-self.close))/((self.high-self.low)).diff()
        #m=result.iloc[-1,:].dropna() 
        #alpha=m[(m<np.inf)&(m>-np.inf)]      
        #return alpha.dropna() 
        return (-1 * delta((((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)), 1))

    ################################################################
    def alpha_003(self):  
        #delay1 = self.close.shift()   
        #condtion1 = (self.close == delay1)
        #condition2 = (self.close > delay1)
        #condition3 = (self.close < delay1)

        #part2 = (self.close-np.minimum(delay1[condition2],self.low[condition2])).iloc[-6:,:] #取最近的6位数据
        #part3 = (self.close-np.maximum(delay1[condition3],self.low[condition3])).iloc[-6:,:] 

        #result=part2.fillna(0)+part3.fillna(0)
        #alpha=result.sum()
        #return alpha.dropna()
        alpha_tmp = np.maximum(self.high, delay(self.close, 1))
        cond = self.close > delay(self.close, 1)
        alpha_tmp[cond] = np.minimum(self.low, delay(self.close, 1))
        alpha = self.close - alpha_tmp
        alpha[self.close == delay(self.close, 1)] = 0
        return ts_sum(alpha, 6)

    ########################################################################
    def alpha_004(self):
        #condition1=(pd.rolling_std(self.close,8)<pd.rolling_sum(self.close,2)/2)
        #condition2=(pd.rolling_sum(self.close,2)/2<(pd.rolling_sum(self.close,8)/8-pd.rolling_std(self.close,8)))
        #condition3=(1<=self.volume/pd.rolling_mean(self.volume,20)) 
        #condition3

        #indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)#[condition2]
        #indicator2=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)#[condition3]

        #part0=pd.rolling_sum(self.close,8)/8
        #part1=indicator2[condition1].fillna(0)
        #part2=(indicator1[~condition1][condition2]).fillna(0)
        #part3=(indicator1[~condition1][~condition2][condition3]).fillna(0)
        #part4=(indicator2[~condition1][~condition2][~condition3]).fillna(0)

        #result=part0+part1+part2+part3+part4
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        cond1 = ((self.volume / sma(self.volume, 20)) >= 1)
        cond11 = ((self.volume / sma(self.volume, 20)) < 1)
        alpha_tmp_tmp = self.close.copy()
        alpha_tmp_tmp[cond1] = 1
        alpha_tmp_tmp[cond11] = -1 * 1 
        cond2 = ts_sum(self.close, 2) / 2 < (ts_sum(self.close, 8) / 8) - stddev(self.close, 8)
        alpha_tmp = alpha_tmp_tmp.copy()
        alpha_tmp[cond2] = 1
        cond3 = ((ts_sum(self.close, 8) / 8) + stddev(self.close, 8)) < (ts_sum(self.close, 2) / 2)
        alpha = alpha_tmp.copy()
        alpha[cond3] = -1 * 1
        return alpha

    ################################################################
    def alpha_005(self):
        #ts_volume=(self.volume.iloc[-7:,:]).rank(axis=0,pct=True)
        #ts_high=(self.high.iloc[-7:,:]).rank(axis=0,pct=True)
        #corr_ts=pd.rolling_corr(ts_high,ts_volume,5) 
        #alpha=corr_ts.max().dropna()
        #alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] 
        #return alpha 
        return (-1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3))

    ###############################################################
    def alpha_006(self):
        #condition1=((self.open_price*0.85+self.high*0.15).diff(4)>1)
        #condition2=((self.open_price*0.85+self.high*0.15).diff(4)==1)
        #condition3=((self.open_price*0.85+self.high*0.15).diff(4)<1)
        #indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns) 
        #indicator2=pd.DataFrame(np.zeros(self.close.shape),index=self.close.index,columns=self.close.columns)
        #indicator3=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns) 
        #part1=indicator1[condition1].fillna(0)
        #part2=indicator2[condition2].fillna(0)
        #part3=indicator3[condition3].fillna(0)
        #result=part1+part2+part3
        #alpha=(result.rank(axis=1,pct=True)).iloc[-1,:]    #cross section rank
        #return alpha.dropna()
        return (rank(sign(delta(((self.open * 0.85) + (self.high * 0.15)), 4))) * -1)


    ##################################################################
    def alpha_007(self):
        #part1=(np.maximum(self.avg_price-self.close,3)).rank(axis=1,pct=True)
        #part2=(np.minimum(self.avg_price-self.close,3)).rank(axis=1,pct=True)
        #part3=(self.volume.diff(3)).rank(axis=1,pct=True)
        #result=part1+part2*part3
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return ((rank(ts_max(self.vwap - self.close, 3)) + rank(ts_min(self.vwap - self.close, 3))) * rank(delta(self.volume, 3)))

    ##################################################################
    def alpha_008(self):
        #temp=(self.high+self.low)*0.2/2+self.avg_price*0.8
        #result=-temp.diff(4)
        #alpha=result.rank(axis=1,pct=True)
        #alpha=alpha.iloc[-1,:]
        #return alpha.dropna()
        return rank(delta(((self.high + self.low) * 0.2 / 2 + (self.vwap * 0.8)), 4) * -1)

    ##################################################################
    def alpha_009(self):
        #temp=(self.high+self.low)*0.5-(self.high.shift()+self.low.shift())*0.5*(self.high-self.low)/self.volume #计算close_{i-1}
        #result=pd.ewma(temp,alpha=2/7)
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        temp = (self.high + self.low) * 0.5 - (delay(self.high, 1) + delay(self.low, 1)) * 0.5 * (self.high - self.low) / self.volume
        #alpha = pd.ewma(temp,alpha=2/7)
        alpha_fin = temp.ewm(alpha=2/7, adjust=False).mean()
        return alpha_fin

    ##################################################################
    def alpha_010(self):
        #ret=self.close.pct_change()
        #condtion=(ret<0)
        #part1=(pd.rolling_std(ret,20)[condtion]).fillna(0)
        #part2=(self.close[~condtion]).fillna(0)
        #result=np.maximum((part1+part2)**2,5)
        #alpha=result.rank(axis=1,pct=True)
        #alpha=alpha.iloc[-1,:]
        #return alpha.dropna()
        #ret = self.close.pct_change()
        ret = self.returns.copy()
        cond = ret < 0 
        alpha = self.close.copy()
        alpha[cond] = stddev(ret, 20)
        return ts_rank(ts_max(alpha.pow(2)), 5)


    ##################################################################
    def alpha_011(self):
        temp=((self.close-self.low)-(self.high-self.close))/(self.high-self.low)
        result=temp*self.volume
        #alpha=result.iloc[-6:,:].sum()
        return ts_sum(result, 6)


    ##################################################################
    def alpha_012(self):
        #vwap10=pd.rolling_sum(self.avg_price,10)/10
        #temp1=self.open_price-vwap10
        #part1=temp1.rank(axis=1,pct=True)
        #temp2=(self.close-self.avg_price).abs()
        #part2=-temp2.rank(axis=1,pct=True)
        #result=part1*part2
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return rank(self.open - (ts_sum(self.vwap, 10) / 10)) * (-1 * rank(abs(self.close - self.vwap)))


    ##################################################################
    def alpha_013(self):
        #result=((self.high-self.low)**0.5)-self.avg_price
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return ((self.high * self.low).pow(0.5) - self.vwap)

    ##################################################################
    def alpha_014(self):
        #result=self.close-self.close.shift(5)
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return self.close - delay(self.close, 5)

    ################################################################## 
    def alpha_015(self):
        #result=self.open_price/self.close.shift()-1
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return self.open / delay(self.close, 1) - 1

    ##################################################################  
    def alpha_016(self):
        #temp1=self.volume.rank(axis=1,pct=True)
        #temp2=self.avg_price.rank(axis=1,pct=True) 
        #part=pd.rolling_corr(temp1,temp2,5)#  
        #part=part[(part<np.inf)&(part>-np.inf)]
        #result=part.iloc[-5:,:]
        #result=result.dropna(axis=1)
        #alpha=-result.max()  
        #return alpha.dropna()
        return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))

    ##################################################################   
    def alpha_017(self):
        #temp1=pd.rolling_max(self.avg_price,15) 
        #temp2=(self.close-temp1).dropna()
        #part1=temp2.rank(axis=1,pct=True)
        #part2=self.close.diff(5)
        #result=part1**part2
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return rank(self.vwap - ts_max(self.vwap, 15)) ** delta(self.close, 5)

    ################################################################## 
    def alpha_018(self):
        #delay5=self.close.shift(5)
        #alpha=self.close/delay5
        #alpha=alpha.iloc[-1,:]
        #return alpha.dropna()
        return self.close / delay(self.close, 5)


    ##################################################################  
    def alpha_019(self):
        #delay5=self.close.shift(5)
        #condition1=(self.close<delay5)
        #condition3=(self.close>delay5)
        #part1=(self.close[condition1]-delay5[condition1])/delay5[condition1]
        #part1=part1.fillna(0)
        #part2=(self.close[condition3]-delay5[condition3])/self.close[condition3]
        #part2=part2.fillna(0)
        #result=part1+part2
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        cond1 = (self.close == delay(self.close, 5))
        temp = (self.close - delay(self.close, 5)) / self.close
        temp[cond1] = 0
        cond2 = self.close < delay(self.close, 5)
        alpha = temp
        alpha[cond2] = (self.close - delay(self.close, 5)) / delay(self.close, 5)
        return alpha

    ##################################################################  
    def alpha_020(self):
        #delay6=self.close.shift(6)
        #result=(self.close-delay6)*100/delay6
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * 100


    ##################################################################   
    def alpha_021(self):
        A = sma(self.close, 6)
        return func_regbert_slope(A, 6)


    ##################################################################    
    def alpha_022(self):
        #part1=(self.close-pd.rolling_mean(self.close,6))/pd.rolling_mean(self.close,6)
        #temp=(self.close-pd.rolling_mean(self.close,6))/pd.rolling_mean(self.close,6)
        #part2=temp.shift(3)
        #result=part1-part2
        #result=pd.ewma(result,alpha=1.0/12)
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()   
        return ((self.close - sma(self.close, 6)) / sma(self.close, 6) - delay((self.close - sma(self.close, 6) / sma(self.close, 6)), 3)).ewm(alpha=1/12, adjust=False, axis=0).mean()

    ##################################################################  
    def alpha_023(self):
        #condition1=(self.close>self.close.shift())
        #temp1=pd.rolling_std(self.close,20)[condition1]
        #temp1=temp1.fillna(0)
        #temp2=pd.rolling_std(self.close,20)[~condition1]
        #temp2=temp2.fillna(0)
        #part1=pd.ewma(temp1,alpha=1.0/20)
        #part2=pd.ewma(temp2,alpha=1.0/20)
        #result=part1*100/(part1+part2)
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        cond1 = self.close > delay(self.close, 1)
        alpha_temp1 = stddev(self.close, 20)
        alpha_temp1[cond1] = 0
        alpha_temp1 = alpha_temp1.ewm(alpha=1/20, adjust=False).mean()
        cond2 = self.close <= delay(self.close, 1)
        alpha_temp2 = stddev(self.close, 20)
        alpha_temp2[cond2] = 0
        alpha_temp2 = alpha_temp2.ewm(alpha=1/20, adjust=False).mean()
        return alpha_temp2 / (alpha_temp1 + alpha_temp2) * 100

    ################################################################## 
    def alpha_024(self):
        #delay5=self.close.shift(5)
        #result=self.close-delay5
        #result=pd.ewma(result,alpha=1.0/5)
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return (self.close - delay(self.close, 5)).ewm(alpha=1/5, adjust=False, axis=0).mean()

    ##################################################################     
    def alpha_025(self):
        #n=9
        #part1=(self.close.diff(7)).rank(axis=1,pct=True)
        #part1=part1.iloc[-1,:]
        #temp=self.volume/pd.rolling_mean(self.volume,20)
        #temp1=temp.iloc[-9:,:]
        #seq=[2*i/(n*(n+1)) for i in range(1,n+1)]   
        #weight=np.array(seq)
        #
        #temp1=temp1.apply(lambda x: x*weight)   
        #ret=self.close.pct_change()   
        #rank_sum_ret=(ret.sum()).rank(pct=True)
        #part2=1-temp1.sum() 
        #part3=1+rank_sum_ret
        #alpha=-part1*part2*part3
        #return alpha.dropna()
        temp = (-1 * rank((delta(self.close, 7)) * (1 - rank(func_decay_linear((self.volume / sma(self.volume, 20)), 9)))))
        alpha = (temp * (1 + rank(ts_sum(self.returns, 250))))
        return alpha


    ##################################################################        
    def alpha_026(self):
        #part1=pd.rolling_sum(self.close,7)/7-self.close
        #part1=part1.iloc[-1,:]
        #delay5=self.close.shift(5)
        #part2=pd.rolling_corr(self.avg_price,delay5,230)
        #part2=part2.iloc[-1,:]
        #alpha=part1+part2
        #return alpha.dropna()
        return (ts_sum(self.close, 7) / 7 - self.close) * (correlation(self.vwap, delay(self.close, 5), 230))


    ##################################################################     
    def alpha_027(self):
        temp1 = (self.close - delay(self.close, 3)) / delay(self.close, 3) * 100 
        temp2 = (self.close - delay(self.close, 6)) / delay(self.close, 6) * 100 
        return func_wma(temp1 + temp2, 12)


    ##################################################################     
    def alpha_028(self):
        #temp1=self.close-pd.rolling_min(self.low,9)
        #temp2=pd.rolling_max(self.high,9)-pd.rolling_min(self.low,9)
        #part1=3*pd.ewma(temp1*100/temp2,alpha=1.0/3)
        #temp3=pd.ewma(temp1*100/temp2,alpha=1.0/3)
        #part2=2*pd.ewma(temp3,alpha=1.0/3)
        #result=part1-part2
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        temp1 = (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100
        temp2 = temp1.ewm(alpha=1/3, adjust=False).mean()
        return 3 * temp2 - 2 * temp2.ewm(alpha=1/3, adjust=False).mean()


    ##################################################################     
    def alpha_029(self):
        #delay6=self.close.shift(6)
        #result=(self.close-delay6)*self.volume/delay6
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * self.volume


    ##################################################################     
    def alpha_030(self):
        
        return 0


    ##################################################################     
    def alpha_031(self):
        #result=(self.close-pd.rolling_mean(self.close,12))*100/pd.rolling_mean(self.close,12)
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return (self.close - sma(self.close, 12)) / sma(self.close, 12) * 100


    ##################################################################     
    def alpha_032(self):
        #temp1=self.high.rank(axis=1,pct=True)
        #temp2=self.volume.rank(axis=1,pct=True)
        #temp3=pd.rolling_corr(temp1,temp2,3)
        #temp3=temp3[(temp3<np.inf)&(temp3>-np.inf)].fillna(0) 
        #result=(temp3.rank(axis=1,pct=True)).iloc[-3:,:]
        #alpha=-result.sum()
        #return alpha.dropna()
        return (-1 * ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3))


    ##################################################################     
    def alpha_033(self):
        #ret=self.close.pct_change()
        #temp1=pd.rolling_min(self.low,5)  #TS_MIN
        #part1=temp1.shift(5)-temp1
        #part1=part1.iloc[-1,:]
        #temp2=(pd.rolling_sum(ret,240)-pd.rolling_sum(ret,20))/220
        #part2=temp2.rank(axis=1,pct=True)
        #part2=part2.iloc[-1,:]
        #temp3=self.volume.iloc[-5:,:]
        #part3=temp3.rank(axis=0,pct=True)   #TS_RANK
        #part3=part3.iloc[-1,:]
        #alpha=part1+part2+part3
        #return alpha.dropna()
        temp1 = -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)
        temp2 = rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
        return temp1 * temp2 * ts_rank(self.volume, 5)


    ##################################################################     
    def alpha_034(self):
        #result=pd.rolling_mean(self.close,12)/self.close
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return sma(self.close, 12) / self.close


    ##################################################################     
    def alpha_035(self):
        #n=15
        #m=7
        #temp1=self.open_price.diff()
        #temp1=temp1.iloc[-n:,:]
        #seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   
        #seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   
        #weight1=np.array(seq1)
        #weight2=np.array(seq2)
        #part1=temp1.apply(lambda x: x*weight1)       
        #part1=part1.rank(axis=1,pct=True)
        #temp2=0.65*self.open_price+0.35*self.open_price
        #temp2=pd.rolling_corr(temp2,self.volume,17)
        #temp2=temp2.iloc[-m:,:]
        #part2=temp2.apply(lambda x: x*weight2)
        #alpha=np.minimum(part1.iloc[-1,:],-part2.iloc[-1,:])
        #alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] 
        #alpha=alpha.dropna()    
        #return alpha
        data1 = rank(func_decay_linear(delta(self.open, 1), 15)) 
        data2 = rank(func_decay_linear(correlation(self.volume, (self.open * 0.65 + self.open * 0.35), 17), 7))
        return np.minimum(data1, data2) * -1


    ##################################################################     
    def alpha_036(self):
        #temp1=self.volume.rank(axis=1,pct=True)
        #temp2=self.avg_price.rank(axis=1,pct=True)
        #part1=pd.rolling_corr(temp1,temp2,6)
        #result=pd.rolling_sum(part1,2)
        #result=result.rank(axis=1,pct=True)
        #alpha=result.iloc[-1,:]
        #return alpha.dropna()
        return rank(ts_sum(correlation(rank(self.volume), rank(self.vwap), 6), 2))


    ##################################################################     
    def alpha_037(self):
        ret = self.returns.copy()
        temp = ts_sum(self.open, 5) * ts_sum(ret, 5)
        part1 = rank(temp)
        part2 = delay(temp, 10)
        result = -part1 - part2
        alpha = result
        return alpha


    ##################################################################     
    def alpha_038(self):
        sum_20 = ts_sum(self.high, 20)/20
        delta2 = delta(self.high, 2)
        cond = (sum_20 >= self.high)
        result = -delta2
        result[cond] = 0
        return result


    ##################################################################     
    def alpha_039(self):
        #n=8
        #m=12
        #temp1=self.close.diff(2)
        #temp1=temp1.iloc[-n:,:]
        #seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]          
        #seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]          
        #weight1=np.array(seq1)
        #weight2=np.array(seq2)
        #part1=temp1.apply(lambda x: x*weight1)       
        #part1=part1.rank(axis=1,pct=True)
        #temp2=0.3*self.avg_price+0.7*self.open_price
        #volume_180=pd.rolling_mean(self.volume,180)
        #sum_vol=pd.rolling_sum(volume_180,37)
        #temp3=pd.rolling_corr(temp2,sum_vol,14)
        #temp3=temp3.iloc[-m:,:]
        #part2=-temp3.apply(lambda x: x*weight2)
        #part2.rank(axis=1,pct=True)
        #result=part1.iloc[-1,:]-part2.iloc[-1,:]
        #alpha=result
        #alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]
        #alpha=alpha.dropna()
        #return alpha
        data1 = rank(func_decay_linear(delta(self.close, 2), 8))
        data2 = rank(func_decay_linear(correlation((self.vwap * 0.3 + self.open * 0.7), ts_sum(sma(self.volume, 180), 37), 14), 12))
        return (data1 - data2) * -1


    ##################################################################     
    def alpha_040(self):
        cond1 = self.close <= delay(self.close, 1)
        cond2 = self.close > delay(self.close, 1)
        data1 = self.volume.copy()
        data1[cond1] = 0
        data2 = self.volume.copy()
        data2[cond2] = 0
        return ts_sum(data1, 26) / ts_sum(data2, 26) * 100


    ##################################################################     
    def alpha_041(self):
        delta_avg = delta(self.vwap, 3)
        part = np.maximum(delta_avg, 5)
        return -rank(part)


    ##################################################################     
    def alpha_042(self):
        part1 = correlation(self.high, self.volume, 10)
        part2 = stddev(self.high, 10)
        part2 = rank(part2)
        result = -part1 * part2
        return result


    ##################################################################     
    def alpha_043(self):
        delay1 = delay(self.close, 1)
        cond1 = (self.close >= delay1)
        cond2 = (self.close > delay1)
        temp1 = -self.volume
        temp1[cond1] = 0 
        temp2 = temp1
        temp2[cond2] = self.volume
        result = ts_sum(temp2,6)
        return result


    ##################################################################     
    def alpha_044(self):
        #part1=self.open_price*0.4+self.close*0.6
        #n=6
        #m=10
        #temp1=pd.rolling_corr(self.low,pd.rolling_mean(self.volume,10),7)
        #temp1=temp1.iloc[-n:,:]
        #seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]          
        #seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]          
        #weight1=np.array(seq1)
        #weight2=np.array(seq2)
        #part1=temp1.apply(lambda x: x*weight1)   
        #part1=part1.iloc[-4:,].rank(axis=0,pct=True)
        #temp2=self.avg_price.diff(3)
        #temp2=temp2.iloc[-m:,:]
        #part2=temp2.apply(lambda x: x*weight2)
        #part2=part1.iloc[-5:,].rank(axis=0,pct=True)
        #alpha=part1.iloc[-1,:]+part2.iloc[-1,:]
        #alpha=alpha.dropna()
        data1 = ts_rank(func_decay_linear(correlation(self.low, sma(self.volume, 10), 7), 6), 4) 
        data2 = ts_rank(func_decay_linear(delta(self.vwap, 3), 10), 15) 
        return data1 + data2


    ##################################################################     
    def alpha_045(self):
        temp1 = self.close * 0.6 + self.open_price * 0.4
        part1 = delta(temp1, 1)
        part1 = rank(part1)
        temp2 = sma(self.volume, 150)
        part2 = correlation(self.vwap, temp2, 15)
        part2 = rank(part2)
        result = part1 * part2
        return result


    ##################################################################     
    def alpha_046(self):
        part1 = sma(self.close, 3)
        part2 = sma(self.close, 6)
        part3 = sma(self.close, 12)
        part4 = sma(self.close, 24)
        result = (part1 + part2 + part3 + part4) * 0.25 / self.close
        return result


    ##################################################################     
    def alpha_047(self):
        part1 = sma(self.high, 6) - self.close
        part2 = ts_max(self.high, 6)- ts_min(self.low, 6)
        result = 100 * part1 / part2
        alpha = result.ewm(alpha=1/9, adjust=False).mean()
        return alpha   


    ##################################################################     
    def alpha_048(self):
        #condition1=(self.close>self.close.shift())
        #condition2=(self.close.shift()>self.close.shift(2))
        #condition3=(self.close.shift(2)>self.close.shift(3))

        #indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition1].fillna(0)
        #indicator2=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition2].fillna(0)
        #indicator3=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition3].fillna(0)

        #indicator11=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition1)&(self.close!=self.close.shift())].fillna(0)
        #indicator22=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition2)&(self.close.shift()!=self.close.shift(2))].fillna(0)
        #indicator33=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition3)&(self.close.shift(2)!=self.close.shift(3))].fillna(0)

        #summ=indicator1+indicator2+indicator3+indicator11+indicator22+indicator33  
        #result=-summ*pd.rolling_sum(self.volume,5)/pd.rolling_sum(self.volume,20)
        #alpha=result.iloc[-1,:].dropna()
        #return alpha
        temp1 = rank(sign(self.close - delay(self.close, 1)) + sign(delay(self.close, 1) - delay(self.close, 2)) + sign(delay(self.close, 2) - delay(self.close, 3))) * ts_sum(self.volume, 5)
        return -1 * temp1 / ts_sum(self.volume, 20)


    ##################################################################     
    def alpha_049(self):
        delay_high = delay(self.high, 1)
        delay_low = delay(self.low, 1)
        cond1 = (self.high + self.low >= delay_high + delay_low)
        cond2 = (self.high + self.low <= delay_high + delay_low)
        part1 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part1[cond1] = 0
        part2 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part2[cond2] = 0
        result = ts_sum(part1, 12) / (ts_sum(part1, 12) + ts_sum(part2, 12))
        return result


    ##################################################################     
    def alpha_050(self):
        delay_high = delay(self.high, 1)
        delay_low = delay(self.low, 1)
        cond1 = (self.high + self.low >= delay_high + delay_low)
        cond2 = (self.high + self.low <= delay_high + delay_low)
        part1 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part1[cond1] = 0
        part2 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part2[cond2] = 0
        result = (ts_sum(part2, 12) - ts_sum(part1, 12)) / (ts_sum(part1, 12) + ts_sum(part2, 12))
        return result


    ##################################################################     
    def alpha_051(self):
        delay_high = delay(self.high, 1)
        delay_low = delay(self.low, 1)
        cond1 = (self.high + self.low >= delay_high + delay_low)
        cond2 = (self.high + self.low <= delay_high + delay_low)
        part1 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part1[cond1] = 0
        part2 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part2[cond2] = 0
        result = ts_sum(part2, 12) / (ts_sum(part1, 12) + ts_sum(part2, 12))
        return result

 
    ##################################################################    
    def alpha_052(self):
        delay_res = delay((self.high + self.low + self.close) / 3, 1)
        part1 = ts_sum(np.maximum(self.high - delay_res, 0), 26)
        part2 = ts_sum(np.maximum(delay_res - self.low, 0), 26)
        alpha = part1 / part2 * 100
        return alpha
   

    ##################################################################    
    def alpha_053(self):
        delay_res = delay(self.close, 1) 
        cond = self.close > delay_res
        alpha = self.close.copy()
        alpha[cond] = 1
        alpha[~cond] = 0
        return ts_sum(alpha, 12) / 12 * 100


    ##################################################################    
    def alpha_054(self):
        #part1=(self.close-self.open_price).abs()
        #part1=part1.std()
        #part2=(self.close-self.open_price).iloc[-1,:]
        #part3=self.close.iloc[-10:,:].corrwith(self.open_price.iloc[-10:,:])
        #result=(part1+part2+part3).dropna()
        #alpha=result.rank(pct=True)
        #return alpha.dropna()
        temp = stddev(np.abs(self.close - self.open) + (self.close - self.open) + correlation(self.close, self.open, 10)) 
        return -1 * rank(temp)
    
    
    ##################################################################    
    def alpha_055(self):
        data1 = 16 * (self.close - delay(self.close, 1) + (self.close - self.open) / 2 + delay(self.close, 1) - delay(self.open, 1))
        cond1 = abs(self.low - delay(self.close, 1)) > abs(self.high - delay(self.low, 1))
        cond2 = abs(self.low - delay(self.close, 1)) > abs(self.high - delay(self.close, 1))
        data2 = abs(self.high - delay(self.low, 1)) + abs(delay(self.close, 1) - delay(self.open, 1)) / 4
        data2[cond1 & cond2] = abs(self.low - delay(self.close, 1)) + abs(self.high - delay(self.close, 1)) / 2 + abs(delay(self.close, 1) - delay(self.open, 1)) / 4
        cond3 = abs(self.high - delay(self.close, 1)) > abs(self.low - delay(self.close, 1))
        cond4 = abs(self.high - delay(self.close, 1)) > abs(self.high - delay(self.low, 1))
        data3 = data2.copy()
        data3[cond3 & cond4] = abs(self.high - delay(self.close, 1)) + abs(self.low - delay(self.close, 1)) / 2 + abs(delay(self.close, 1) - delay(self.open, 1)) / 4
        data4 = np.maximum(abs(self.high - delay(self.close, 1)), abs(self.low - delay(self.close, 1)))  
        alpha = ts_sum(data1 / data3 * data4, 20)
        return alpha


    ##################################################################    
    def alpha_056(self):
        #part1=self.open_price.iloc[-1,:]-self.open_price.iloc[-12:,:].min()
        #part1=part1.rank(pct=1)
        #temp1=(self.high+self.low)/2
        #temp1=pd.rolling_sum(temp1,19)
        #temp2=pd.rolling_sum(pd.rolling_mean(self.volume,40),19)
        #part2=temp1.iloc[-13:,:].corrwith(temp2.iloc[-13:,:])
        #part2=(part2.rank(pct=1))**5
        #part2=part2.rank(pct=1)
        #part1[part1<part2]=1                        
        #part1=part1.apply(lambda x: 0 if x <1 else None)
        #alpha=part1.fillna(1)
        #return alpha.dropna()
        data1 = rank(self.open - ts_min(self.open, 12))
        data2 = rank(rank(correlation(ts_sum((self.high + self.low) / 2, 19), ts_sum(sma(self.volume, 40), 19), 13) ** 5))
        cond = data1 < data2
        return cond

    ##################################################################    
    def alpha_057(self):
        part1 = self.close - ts_min(self.low, 9)
        part2 = ts_max(self.high, 9) - ts_min(self.low, 9)
        result = (100 * part1 / part2).ewm(alpha=1.0/3, adjust=False).mean()
        alpha = result
        return alpha

    ##################################################################    
    def alpha_058(self):
        delay_res = delay(self.close, 1) 
        cond = self.close > delay_res
        alpha = self.close.copy()
        alpha[cond] = 1
        alpha[~cond] = 0
        return ts_sum(alpha, 20) / 20 * 100

    ##################################################################    
    def alpha_059(self):
        delay_res = delay(self.close, 1)
        cond1 = (self.close > delay_res)
        cond2 = (self.close == delay_res)
        part1 = np.maximum(self.high, delay_res)
        part1[cond1] = np.minimum(self.low, delay_res)
        part2 = self.close - part1
        part2[cond2] = 0
        return ts_sum(part2, 20)

    ##################################################################    
    def alpha_060(self):
        part1 = (self.close - self.low) - (self.high - self.close)
        part2 = self.high - self.low
        result = self.volume * part1 / part2
        alpha = ts_sum(result, 20)
        return alpha


    ##################################################################    
    def alpha_061(self):
        #n=12
        #m=17
        #temp1=self.avg_price.diff()
        #temp1=temp1.iloc[-n:,:]
        #seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]          
        #seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]          
        #weight1=np.array(seq1)
        #weight2=np.array(seq2)
        #part1=temp1.apply(lambda x: x*weight1)       
        #part1=part1.rank(axis=1,pct=True)
        #temp2=self.low
        #temp2=pd.rolling_corr(temp2,pd.rolling_mean(self.volume,80),8)
        #temp2=temp2.rank(axis=1,pct=1)
        #temp2=temp2.iloc[-m:,:]
        #part2=temp2.apply(lambda x: x*weight2)
        #part2=-part2.rank(axis=1,pct=1)
        #alpha=np.maximum(part1.iloc[-1,:],part2.iloc[-1,:])
        #alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]    
        #alpha=alpha.dropna()    
        data1 = rank(func_decay_linear(delta(self.vwap, 1), 12))
        data2 = rank(func_decay_linear(rank(correlation(self.low, sma(self.volume, 80), 8)), 17))
        return np.maximum(data1, data2) * -1


    ##################################################################    
    def alpha_062(self):
        #volume_rank=self.volume.rank(axis=1,pct=1)
        #result=self.high.iloc[-5:,:].corrwith(volume_rank.iloc[-5:,:])
        #alpha=-result
        #return alpha.dropna() 
        return -1 * correlation(self.high, rank(self.volume), 5)


    ##################################################################    
    def alpha_063(self):
        part1 = np.maximum(self.close - delay(self.close, 1), 0)
        part1 = part1.ewm(alpha=1.0/6, adjust=False).mean()
        part2 = np.abs(self.close - delay(self.close, 1))
        part2 = part2.ewm(alpha=1.0/6, adjust=False).mean()
        result = part1 * 100 / part2
        return result


    ##################################################################    
    def alpha_064(self):
        #n=4
        #m=14
        #temp1=pd.rolling_corr(self.avg_price.rank(axis=1,pct=1),self.volume.rank(axis=1,pct=1),4)
        #temp1=temp1.iloc[-n:,:]
        #seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]       
        #seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]        
        #weight1=np.array(seq1)
        #weight2=np.array(seq2)
        #part1=temp1.apply(lambda x: x*weight1)       
        #part1=part1.rank(axis=1,pct=True)
        #temp2=self.close.rank(axis=1,pct=1)
        #temp2=pd.rolling_corr(temp2,pd.rolling_mean(self.volume,60),4)
        #temp2=np.maximum(temp2,13)
        #temp2=temp2.iloc[-m:,:]
        #part2=temp2.apply(lambda x: x*weight2)
        #part2=-part2.rank(axis=1,pct=1)
        #alpha=np.maximum(part1.iloc[-1,:],part2.iloc[-1,:])
        #alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]    
        #alpha=alpha.dropna()    
        data1 = rank(func_decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4))
        data2 = rank(func_decay_linear(np.maximum(correlation(rank(self.close), rank(sma(self.volume, 60)), 4), 13), 14))
        return np.maximum(data1, data2) * -1


    ##################################################################    
    def alpha_065(self):
        part1 = self.close.copy()
        alpha = sma(part1, 6) / self.close
        return alpha


    ##################################################################    
    def alpha_066(self):
        part1 = self.close.copy()
        alpha = (self.close - sma(part1, 6)) * 100 / sma(part1, 6)
        return alpha


    ##################################################################    
    def alpha_067(self):
        temp1 = self.close - delay(self.close, 1)
        part1 = np.maximum(temp1,0)
        part1 = part1.ewm(alpha=1.0/24, adjust=False).mean()
        temp2 = abs(temp1)
        part2 = temp2.ewm(alpha=1.0/24, adjust=False).mean()
        result = part1 / part2 * 100
        return result


    ##################################################################    
    def alpha_068(self):
        part1 = (self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2
        part2 = (self.high - self.low) / self.volume
        result = part1 * part2
        result = result.ewm(alpha=2.0/15, adjust=False).mean()
        return result
    
    
    ##################################################################
    def alpha_069(self):
        cond1 = self.open <= delay(self.open, 1)
        DTM = np.maximum(self.high - self.open, self.open - delay(self.open, 1))
        DTM[cond1] = 0
        cond2 = self.open >= delay(self.open, 1)
        DBM = np.maximum(self.open - self.low, self.open - delay(self.open, 1))
        cond3 = ts_sum(DTM, 20) == ts_sum(DBM, 20)
        temp1 = (ts_sum(DTM, 20) - ts_sum(DBM, 20)) / ts_sum(DBM, 20)
        temp1[cond3] = 0
        cond4 = ts_sum(DTM, 20) > ts_sum(DBM, 20)
        alpha = temp1.copy()
        alpha[cond4] = (ts_sum(DTM, 20) - ts_sum(DBM, 20)) / ts_sum(DTM, 20)
        return alpha
    
    
    ##################################################################
    def alpha_070(self):
    #### STD(AMOUNT, 6)
    ##       
        alpha = stddev(self.amount, 6)
        return alpha
    
    
    #############################################################################
    def alpha_071(self):
    # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100 
    #     
        data = (self.close - sma(self.close, 24)) / sma(self.close, 24)
        alpha = data * 100
        return alpha
    
    
    #############################################################################
    def alpha_072(self):
    #SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    #     
        data1 = ts_max(self.high, 6) - self.close
        data2 = ts_max(self.high, 6) - ts_min(self.low, 6)
        alpha = (data1 / data2 * 100).ewm(alpha=1/15, adjust=False).mean()
        return alpha
    
    
    #############################################################################
    def alpha_073(self):
        data1 = ts_rank(func_decay_linear(func_decay_linear(correlation(self.close, self.volume, 10), 16), 4), 5) 
        data2 = rank(func_decay_linear(correlation(self.vwap, sma(self.volume, 30), 4), 3))
        return (data1 - data2) * -1
        
        
    #############################################################################    
    def alpha_074(self):
    #(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))) 
    #     
        data1 = ts_sum((self.low * 0.35 + self.vwap * 0.65), 20)
        data2 = ts_sum(sma(self.volume, 40), 20)
        rank1 = correlation(data1, data2, 7)
        data3 = rank(self.vwap)
        data4 = rank(self.volume)
        rank2 = rank(correlation(data3, data4, 6))
        alpha = (rank1 + rank2)
        return alpha
    
    
    #############################################################################
    def alpha_075(self):
    #COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50) 
    #     
        #benchmark = get_price('000300.XSHG', None, end_date, '1d', ['open','close'], False, None, 50)
        #condition = benchmark['close'] < benchmark['open']
	#data1 = benchmark[condition]
        #numbench = len(data1)
        #timelist = data1.index.tolist()
        #data2 = pd.merge(self.close, data1, left_index=True, right_index=True).drop(['close', 'open'], axis=1)
        #data3 = pd.merge(self.open_price, data1, left_index=True, right_index=True).drop(['close', 'open'], axis=1)
        #data4 = data2[data2 > data3]
        #alpha = 1 - data4.isnull().sum(axis=0) / numbench
        cond1 = self.close > self.open
        cond2 = self.benchmark_close < self.benchmark_open
        data1 = self.close.copy()
        data1[cond1 & cond2] = 1
        data1[~(cond1 & cond2)] = 0
        data2 = self.close.copy()
        data2[cond2] = 1
        data2[~cond2] = 0
        return ts_sum(data1, 50) / ts_sum(data2, 50)
    
    
    #############################################################################
    def alpha_076(self):
    #STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20) 
    #     
        data1 = stddev(abs((self.close / delay(self.close, 1) - 1)) / self.volume, 20)
        data2 = sma(abs((self.close / delay(self.prev_close, 1) - 1)) / self.volume, 20)
        alpha = (data1 / data2)
        return alpha
    
    
    #############################################################################
    def alpha_077(self):
    #MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))     
        #data1 = ((self.high + self.low) / 2 + self.high - (self.avg_price + self.high)).iloc[-20:,:]
        #decay_weights = np.arange(1,20+1,1)[::-1]
        #decay_weights = decay_weights / decay_weights.sum()
        #rank1 = data1.apply(lambda x : x * decay_weights).rank(axis=1, pct=True)
        #data2 = pd.rolling_corr((self.high + self.low)/2, pd.rolling_mean(self.volume, window=40), window=3).iloc[-6:,:]
        #decay_weights2 = np.arange(1,6+1,1)[::-1]
        #decay_weights2 = decay_weights2 / decay_weights2.sum()
        #rank2 = data2.apply(lambda x : x * decay_weights2).rank(axis=1, pct=True)
        #alpha = np.minimum(rank1.iloc[-1], rank2.iloc[-1])
        data1 = rank(func_decay_linear((self.high + self.low) / 2 + self.high - (self.vwap + self.high), 20))
        data2 = rank(func_decay_linear(correlation((self.high + self.low) / 2, sma(self.volume, 40), 3), 6))
        return np.minimum(data1, data2)
    
    #############################################################################
    def alpha_078(self):
    #((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12)) 
    #     
        data1 = (self.high + self.low + self.close) / 3 - sma((self.high + self.low + self.close) / 3, 12)
        data2 = abs(self.close - sma((self.high + self.low + self.close) / 3, 12))
        data3 = sma(data2, 12) * 0.015
        alpha = (data1 / data3)  
        return alpha
    
    
    #############################################################################
    def alpha_079(self):
    #SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    #     
        data1 = (np.maximum((self.close - delay(self.close, 1)), 0)).ewm(alpha=1/12, adjust=False).mean()
        data2 = abs(self.close - delay(self.close, 1)).ewm(alpha=1/12, adjust=False).mean()
        alpha = (data1 / data2 * 100)
        return alpha
    
    
    #############################################################################
    def alpha_080(self):
    #(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    #     
        alpha =  ((self.volume - delay(self.volume, 5))/ delay(self.volume, 5) * 100)
        return alpha
    
    
    #############################################################################
    def alpha_081(self):
        result = self.volume.ewm(alpha=2.0/21, adjust=False).mean()
        return result

    
    #############################################################################
    def alpha_082(self):
        part1 = ts_max(self.high, 6) - self.close
        part2 = ts_max(self.high, 6) - ts_min(self.low, 6)
        result = (100*part1/part2).ewm(alpha=1.0/20, adjust=False).mean()
        alpha=result
        return alpha
  

    #############################################################################
    def alpha_083(self):
        #part1=self.high.rank(axis=0,pct=True) 
        #part1=part1.iloc[-5:,:]
        #part2=self.volume.rank(axis=0,pct=True) 
        #part2=part2.iloc[-5:,:]
        #result=part1.corrwith(part2)
        #alpha=-result
        #return alpha.dropna()
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))


    #############################################################################
    def alpha_084(self):
        cond1 = (self.close >= delay(self.close, 1))
        cond2 = (self.close > delay(self.close, 1))
        part1 = -self.volume.copy()
        part1[cond1] = 0
        part2 = part1.copy()
        part2[cond2] = self.volume.copy()
        alpha = ts_sum(part2, 20)
        return alpha
    
    
    #############################################################################
    def alpha_085(self):
        #temp1=self.volume.iloc[-20:,:]/self.volume.iloc[-20:,:].mean() 
        #temp1=temp1 
        #part1=temp1.rank(axis=0,pct=True)
        #part1=part1.iloc[-1,:] 

        #delta=self.close.diff(7)
        #temp2=-delta.iloc[-8:,:]
        #part2=temp2.rank(axis=0,pct=True).iloc[-1,:]
        #part2=part2 
        #alpha=part1*part2
        #return alpha.dropna()
        part1 = ts_rank(self.volume / sma(self.volume, 20), 20)
        part2 = ts_rank(-1 * delta(self.close, 7), 8)
        return part1 * part2
    
    
    #############################################################################
    def alpha_086(self):
        delay10 = delay(self.close, 10)
        delay20 = delay(self.close, 20)
        temp = (delay20 - delay10) / 10 - (delay10 - self.close) / 10
        cond1 = (temp > 0.25)
        cond2 = (temp < 0)
        temp2 = -1 * (self.close - delay(self.close, 1))
        temp2[cond2] = 1
        alpha = temp2.copy()
        alpha[cond1] = -1
        return alpha
    
    
    #############################################################################
    def alpha_087(self):
        #n=7
        #m=11
        #temp1=self.avg_price.diff(4)
        #temp1=temp1.iloc[-n:,:]
        #seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]       
        #seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]        
        #weight1=np.array(seq1)
        #weight2=np.array(seq2)
        #part1=temp1.apply(lambda x: x*weight1)       
        #part1=part1.rank(axis=1,pct=True)
        #temp2=self.low-self.avg_price
        #temp3=self.open_price-0.5*(self.high+self.low)
        #temp2=temp2/temp3
        #temp2=temp2.iloc[-m:,:]
        #part2=-temp2.apply(lambda x: x*weight2)
        #part2=part2.rank(axis=0,pct=1)
        #alpha=part1.iloc[-1,:]+part2.iloc[-1,:]
        #alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]    
        #alpha=alpha.dropna()    
        #return alpha
        data1 = rank(func_decay_linear(delta(self.vwap, 4), 7))
        data2 = ts_rank(func_decay_linear((self.low - self.vwap) / (self.open - (self.high + self.low) /2), 11), 7)
        return (data1 + data2) * -1
    
    '''
    ########################################################################
    '''
    def alpha_088(self):
        #(close-delay(close,20))/delay(close,20)*100
        ####################      
        #data1=self.close.iloc[-21,:]
        #alpha=((self.close.iloc[-1,:]-data1)/data1)*100
        #alpha=alpha.dropna()
        #return alpha
        return (self.close - delay(self.close, 20)) / delay(self.close, 20) * 100
    
    def alpha_089(self):
        #2*(sma(close,13,2)-sma(close,27,2)-sma(sma(close,13,2)-sma(close,27,2),10,2))
        ######################      
        data1 = (self.close).ewm(alpha=2/13, adjust=False).mean()
        data2 = (self.close).ewm(alpha=2/27, adjust=False).mean()
        data3 = (data1 - data2).ewm(alpha=2/10, adjust=False).mean()
        alpha = 2 * (data1 - data2 - data3)
        return alpha
    
    
    def alpha_090(self):
        #(rank(corr(rank(vwap),rank(volume),5))*-1)
        #######################      
        #data1=self.avg_price.rank(axis=1,pct=True)
        #data2=self.volume.rank(axis=1,pct=True)
        #corr=data1.iloc[-5:,:].corrwith(data2.iloc[-5:,:])
        #rank1=corr.rank(pct=True)
        #alpha=rank1*-1
        #alpha=alpha.dropna()
        #return alpha
        return -1 * rank(correlation(rank(self.vwap), rank(self.volume), 5)) 
    
    def alpha_091(self):
        #((rank((close-max(close,5)))*rank(corr((mean(volume,40)),low,5)))*-1)
        #################      
        #data1=self.close
        #cond=data1>5
        #data1[~cond]=5
        #rank1=((self.close-data1).rank(axis=1,pct=True)).iloc[-1,:]
        #mean=pd.rolling_mean(self.volume,window=40)
        #corr=mean.iloc[-5:,:].corrwith(self.low.iloc[-5:,:])
        #rank2=corr.rank(pct=True) 
        #alpha=rank1*rank2*(-1)
        #alpha=alpha.dropna()
        #return alpha
        return -1 * rank(self.close - ts_max(self.close, 5)) * rank(correlation(sma(self.volume, 40), self.low, 5)) 
    
    #       
    def alpha_092(self):
        # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1) #
        #delta = (self.close * 0.35 + self.vwap * 0.65)-(self.close*0.35+self.avg_price*0.65).shift(2)
        #rank1 = (pd.rolling_apply(delta, 3, self.func_decaylinear)).rank(axis=1, pct=True)
        #rank2 = pd.rolling_apply(pd.rolling_apply(self.volume.rolling(180).mean().rolling(13).corr(self.close).abs(), 5, self.func_decaylinear), 15, self.func_rank)
        #cond_max = rank1>rank2
        #rank2[cond_max] = rank1[cond_max]
        #alpha = (-rank2).iloc[-1,:]
        #alpha=alpha.dropna()
        data1 = rank(func_decay_linear(delta(self.close * 0.35 + self.vwap * 0.65, 2), 3))
        data2 = ts_rank(func_decay_linear(abs(correlation(sma(self.volume, 180), self.close, 13)), 5), 15)
        return np.maximum(data1, data2) * -1
    
    
    #       
    def alpha_093(self):
        # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20) #
        cond = self.open >= delay(self.open, 1)
        data1 = self.open - self.low
        data2 = self.open - delay(self.open, 1)
        alpha = np.maximum(data1, data2)
        alpha[cond] = 0
        return ts_sum(alpha, 20)
    
    
    #       
    def alpha_094(self):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30) #
        cond1 = self.close >= delay(self.close, 1)
        cond2 = self.close > delay(self.close, 1)
        value = -self.volume.copy()
        value[cond1] = 0
        alpha = value.copy()
        alpha[cond2] = self.volume.copy()
        return ts_sum(alpha, 30)
    
    #       
    def alpha_095(self):
        # STD(AMOUNT,20) #
        #alpha = self.amount.iloc[-20:,:].std()
        #alpha=alpha.dropna()
        return stddev(self.amount, 20)
    
    #       
    def alpha_096(self):
        # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1) #
        sma1 = 100 * (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)).ewm(alpha=1/3, adjust=False).mean()
        result = sma1.ewm(alpha=1/3, adjust=False).mean()
        return result
    
    #       
    def alpha_097(self):
        # STD(VOLUME,10) #
        #alpha = self.volume.iloc[-10:,:].std()
        #alpha=alpha.dropna()
        return stddev(self.volume, 10)
    
    #       
    def alpha_098(self):
        # ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)||((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))) #
        sum_close = ts_sum(self.close, 100)
        cond = (delta((sum_close/100), 100)) / delay(self.close, 100) <= 0.05
        alpha = -delta(self.close, 3)
        alpha[cond] = -(self.close - ts_min(self.close, 100))
        return alpha
    
    #       
    def alpha_099(self):
        # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))) #
        #alpha = (-pd.rolling_cov(self.close.rank(axis=1, pct=True), self.volume.rank(axis=1, pct=True), window=5).rank(axis=1, pct=True)).iloc[-1,:]
        #alpha=alpha.dropna()
        return -rank(covariance(rank(self.close), rank(self.volume), 5))
    
    #       
    def alpha_100(self):
        # STD(VOLUME,20) #
        #alpha = self.volume.iloc[-20:,:].std()
        #alpha=alpha.dropna()
        return stddev(self.volume, 20)
    
    #       
    def alpha_101(self):
        # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1) #
        #rank1 = (self.close.rolling(window=15).corr((self.volume.rolling(window=30).mean()).rolling(window=37).sum())).rank(axis=1, pct=True)
        #rank2 = (self.high*0.1+self.avg_price*0.9).rank(axis=1, pct=True)
        #rank3 = self.volume.rank(axis=1, pct=True)
        #rank4 = (rank2.rolling(window=11).corr(rank3)).rank(axis=1, pct=True)
        #alpha = -(rank1<rank4)
        #alpha=alpha.iloc[-1,:].dropna()
        data1 = rank(correlation(self.close, ts_sum(sma(self.volume, 30), 37), 15))
        data2 = rank(correlation(rank(self.high * 0.1 + self.vwap * 0.9), rank(self.volume), 11))
        return -1 * (data1 < data2)
    
    #       
    def alpha_102(self):
        # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100 #
        data = self.volume - delay(self.volume, 1)
        max_data = (np.maximum(data, 0)).ewm(alpha=1/6, adjust=False).mean()
        abs_data = (np.abs(data)).ewm(alpha=1/6, adjust=False).mean()
        return max_data / abs_data * 100
    
    def alpha_103(self):
        ##### ((20-LOWDAY(LOW,20))/20)*100 
        #alpha = (20 - self.low.iloc[-20:,:].apply(self.func_lowday))/20*100
        #alpha=alpha.dropna()
        return (20 - func_low_day(self.low, 20)) / 20 * 100
    
    #       
    def alpha_104(self):
        # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20)))) #
        temp = delta(correlation(self.high, self.volume, 5), 5)
        alpha = (-1 * temp * rank(stddev(self.close, 20)))
        return alpha
    
    
    #       
    def alpha_105(self):
        # (-1*CORR(RANK(OPEN),RANK(VOLUME),10)) #
        #alpha = -((self.open_price.rank(axis=1, pct=True)).iloc[-10:,:]).corrwith(self.volume.iloc[-10:,:].rank(axis=1, pct=True))
        #alpha=alpha.dropna()
        return -correlation(rank(self.open), rank(self.volume), 10)
    
    
    #       
    def alpha_106(self):
        # CLOSE-DELAY(CLOSE,20) #
        #alpha = (self.close-self.close.shift(20)).iloc[-1,:]
        #alpha=alpha.dropna()
        return self.close - delay(self.close, 20)
    
    
    #       
    def alpha_107(self):
        # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1)))) #
        rank1 = -rank(self.open - delay(self.high, 1))
        rank2 = rank(self.open - delay(self.close, 1))
        rank3 = rank(self.open - delay(self.low, 1))
        alpha = (rank1 * rank2 * rank3)
        return alpha
    
    
    #       
    def alpha_108(self):
        # ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1) #
        data = ts_min(self.high, 2)
        rank1 = rank(self.high - data)
        rank2 = rank(correlation(self.vwap, sma(self.volume, 120), 6))
        alpha = (-rank1**rank2)
        return alpha
    
    #       
    def alpha_109(self):
        # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
        data = self.high - self.low
        sma1 = data.ewm(alpha=2/10, adjust=False).mean()
        sma2 = sma1.ewm(alpha=2/10, adjust=False).mean()
        alpha = (sma1/sma2)
        return alpha
    
    
    #       
    def alpha_110(self):
        # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100 #
        data1 = np.maximum(self.high - delay(self.close, 1), 0)
        data2 = np.maximum(delay(self.close, 1) - self.low, 0)
        alpha = ts_sum(data1, 20) / ts_sum(data2, 20) * 100
        return alpha
    
    
    def alpha_111(self):
        #sma(vol*((close-low)-(high-close))/(high-low),11,2)-sma(vol*((close-low)-(high-close))/(high-low),4,2)
        ######################      
        data1 = self.volume * ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        x = data1.ewm(alpha=2/11, adjust=False).mean()
        y = data1.ewm(alpha=2/4, adjust=False).mean()
        alpha = (x-y)
        return alpha
    
    
    #       
    def alpha_112(self):
        # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100 #
        data1 = np.maximum(self.close - delay(self.close, 1), 0)
        data2 = np.minimum(self.close - delay(self.close, 1), 0)
        data2 = np.abs(data2)
        sum1 = ts_sum(data1, 12)
        sum2 = ts_sum(data2, 12)
        alpha = ((sum1 - sum2) / (sum1 + sum2)*100)
        return alpha
    
    
    def alpha_113(self):
        #(-1*((rank((sum(delay(close,5),20)/20))*corr(close,volume,2))*rank(corr(sum(close,5),sum(close,20),2))))
        #####################      
        data1 = rank(ts_sum(delay(self.close, 5) , 20) / 20)
        data2 = correlation(self.close, self.volume, 2)
        data3 = rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        alpha = (-1 * data1 * data2 * data3)
        return alpha
    
    
    def alpha_114(self):
        #((rank(delay(((high-low)/(sum(close,5)/5)),2))*rank(rank(volume)))/(((high-low)/(sum(close,5)/5))/(vwap-close)))
        #####################      
        data1 = rank(delay((self.high - self.low) / (ts_sum(self.close, 5) / 5), 2))
        data2 = rank(rank(self.volume))
        data3 = (((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close))
        alpha = (data1 * data2) / data3
        return alpha 
    
    #       
    def alpha_115(self):
        # RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)) #
        data1 = (self.high * 0.9 + self.close * 0.1)
        data2 = sma(self.volume, 30)
        part1 = rank(correlation(data1, data2, 10))
        tsrank1 = ts_rank((self.high+self.low)/2, 4)
        part2 = rank(correlation(tsrank1, ts_rank(self.volume, 10), 7))
        alpha = part1 ** part2
        return alpha
    
    
    #       
    def alpha_116(self):
        # REGBETA(CLOSE,SEQUENCE,20) #
        alpha = func_regbert_slope(self.close, 20)
        return alpha

    
    def alpha_117(self):
        #######((tsrank(volume,32)*(1-tsrank(((close+high)-low),16)))*(1-tsrank(ret,32)))
        ####################      
        data1 = 1- ts_rank(self.close + self.high - self.low, 16)
        alpha = ts_rank(self.volume, 32) * data1 * (1 - ts_rank(self.returns, 32))
        return alpha 
    
    
    def alpha_118(self):
        ######sum(high-open,20)/sum((open-low),20)*100
        ###################      
        data1 = ts_sum(self.high - self.open, 20)
        data2 = ts_sum(self.open - self.low, 20)
        alpha = ((data1 / data2) * 100)
        return alpha
    
    
    #        
    def alpha_119(self):
        # (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
        #sum1 = (self.volume.rolling(window=5).mean()).rolling(window=26).sum()
        #corr1 = self.avg_price.rolling(window=5).corr(sum1)
        #rank1 = pd.rolling_apply(corr1, 7, self.func_decaylinear).rank(axis=1, pct=True)
        #rank2 = self.open_price.rank(axis=1, pct=True)
        #rank3 = (self.volume.rolling(window=15).mean()).rank(axis=1, pct=True)
        #rank4 = pd.rolling_apply(rank2.rolling(window=21).corr(rank3).rolling(window=9).min(), 7, self.func_rank)
        #rank5 = pd.rolling_apply(rank4, 8, self.func_decaylinear).rank(axis=1, pct=True)
        #alpha = (rank1 - rank5).iloc[-1,:]
        #alpha=alpha.dropna()
        data1 = rank(func_decay_linear(correlation(self.vwap, ts_sum(sma(self.volume, 5), 26), 5), 7))
        data2 = rank(func_decay_linear(ts_rank(np.minimum(correlation(rank(self.open), rank(sma(self.volume, 15)), 21), 9), 7), 8))
        return data1 - data2
    
    
    def alpha_120(self):
        ###############(rank(vwap-close))/(rank(vwap+close))
        ###################      
        data1 = rank(self.vwap - self.close)
        data2 = rank(self.vwap + self.close)
        alpha = (data1 / data2)
        return alpha
    
    
    def alpha_121(self):
        part1 = rank(self.vwap - ts_min(self.vwap, 12))
        part2 = ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(sma(self.volume, 60), 2), 18), 3)
        return -1 * part1 ** part2
    
    
    def alpha_122(self):
        ##### (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) - DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)) 
        ##### / DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
        data = log(self.close)
        part1 = ((data.ewm(alpha=2/13, adjust=False).mean()).ewm(alpha=2/13, adjust=False).mean()).ewm(alpha=2/13, adjust=False).mean()
        part2 = delay(part1, 1)
        return (part1 - part2) / part2
        
        
    def alpha_123(self):
        #####((RANK(CORR(SUM(((HIGH+LOW)/2), 20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
        #data1 = ts_sum((self.high + self.low) / 2, 20)
        #data2 = ts_sum(sma(self.volume, 60), 20)
        #rank1 = data1.iloc[-9:,:].corrwith(data2.iloc[-9:,:]).dropna().rank(axis=0, pct=True)
        #rank2 = self.low.iloc[-6:,:].corrwith(self.volume.iloc[-6:,:]).dropna().rank(axis=0, pct=True)
        #rank1 = rank1[rank1.index.isin(rank2.index)]
        #rank2 = rank2[rank2.index.isin(rank1.index)]
        #alpha = (rank1 < rank2) * (-1)
        #alpha=alpha.dropna()
        data1 = rank(correlation(ts_sum((self.high + self.low) / 2, 20), ts_sum(sma(self.volume, 60), 20), 9)) 
        data2 = rank(correlation(self.low, self.volume, 6))
        return (data1 < data2) * -1
    
    
    def alpha_124(self):
        ##### (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
        ##       
        #data1 = self.close.rolling(30).max().rank(axis=1, pct=True)
        #alpha = (self.close.iloc[-1,:] - self.avg_price.iloc[-1,:]) / (2./3*data1.iloc[-2,:] + 1./3*data1.iloc[-1,:])
        #alpha=alpha.dropna()
        data1 = self.close - self.vwap
        data2 = func_decay_linear(rank(ts_max(self.close, 30)), 2)
        return data1 / data2
    
    
    def alpha_125(self):
        ##### (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME, 80), 17), 20)) / RANK(DECAYLINEAR(DELTA((CLOSE * 0.5 + VWAP * 0.5), 3), 16)))
        #data1 = pd.rolling_corr(self.avg_price, self.volume.rolling(80).mean(), window = 17)
        #decay_weights = np.arange(1,21,1)[::-1]    # 倒序数组
        #decay_weights = decay_weights / decay_weights.sum()
        #rank1 = data1.iloc[-20:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        #data2 = (self.close * 0.5 + self.avg_price * 0.5).diff(3)
        #decay_weights = np.arange(1,17,1)[::-1]    # 倒序数组
        #decay_weights = decay_weights / decay_weights.sum()
        #rank2 = data2.iloc[-16:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        #alpha = rank1 / rank2
        #alpha=alpha.dropna()
        data1 = rank(func_decay_linear(correlation(self.vwap, sma(self.volume, 80), 17), 20))
        data2 = rank(func_decay_linear(delta(self.close * 0.5 + self.vwap * 0.5, 3), 16))
        return data1 / data2
    
    
    def alpha_126(self):
        #### (CLOSE + HIGH + LOW) / 3
        alpha = (self.close + self.high + self.low) / 3
        return alpha
    
    def alpha_127(self):
        return
    
    
    def alpha_128(self):
        temp = (self.high + self.low + self.close) / 3 
        cond1 = (temp <= delay(temp, 1))
        part1 = temp * self.volume
        part1[cond1] = 0
        cond2 = (temp >= delay(temp, 1))
        part2 = temp * self.volume
        part2[cond2] = 0
        alpha = 100 - (100/ ((1 + ts_sum(part1, 14)) / ts_sum(part2, 14)))
        return alpha
    
    
    def alpha_129(self):
        #### SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)):0), 12)
        ##       
        data = self.close - delay(self.close, 1)
        cond = data >= 0
        temp = abs(data)
        data[cond] = 0
        return ts_sum(data, 12)

    
    def alpha_130(self):
        #### alpha_130
        #### (RANK(DELCAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) / RANK(DELCAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))
        #data1 = (self.high + self.low) / 2
        #data2 = self.volume.rolling(40).mean()
        #data3 = pd.rolling_corr(data1, data2, window=9)
        #decay_weights = np.arange(1,11,1)[::-1]    # 倒序数组
        #decay_weights = decay_weights / decay_weights.sum()
        #rank1 = data3.iloc[-10:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)
        #data1 = self.avg_price.rank(axis=1, pct=True)
        #data2 = self.volume.rank(axis=1, pct=True)
        #data3 = pd.rolling_corr(data1, data2, window=7)
        #decay_weights = np.arange(1,4,1)[::-1]    # 倒序数组
        #decay_weights = decay_weights / decay_weights.sum()
        #rank2 = data3.iloc[-3:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)
        #alpha = (rank1 / rank2).dropna()
        data1 = rank(func_decay_linear(correlation((self.high + self.low) / 2, sma(self.volume, 40), 9), 10))
        data2 = rank(func_decay_linear(correlation(rank(self.vwap), rank(self.volume), 7), 3))
        return data1 / data2
    
    def alpha_131(self):
        temp1 = rank(delta(self.vwap, 1))
        temp2 = ts_rank(correlation(self.close, sma(self.volume, 50), 18), 18)
        return temp1 ** temp2
    
    def alpha_132(self):
        #### MEAN(AMOUNT, 20)
        #alpha = self.amount.iloc[-20:,:].mean()
        #alpha=alpha.dropna()
        #return alpha
        return sma(self.amount, 20)
    
    
    def alpha_133(self):
        #### alpha_133
        #### ((20 - HIGHDAY(HIGH, 20)) / 20)*100 - ((20 - LOWDAY(LOW, 20)) / 20)*100
        #alpha = (20 - self.high.iloc[-20:,:].apply(self.func_highday))/20*100 \
        #         - (20 - self.low.iloc[-20:,:].apply(self.func_lowday))/20*100
        #alpha=alpha.dropna()
        data1 = (20 - func_high_day(self.high, 20)) / 20 * 100
        data2 = (20 - func_low_day(self.low, 20)) / 20 * 100
        return data1 - data2
    
    def alpha_134(self):
        #### (CLOSE - DELAY(CLOSE, 12)) / DELAY(CLOSE, 12) * VOLUME
        part = self.close - delay(self.close, 12)
        alpha = part / delay(self.close, 12) * self.volume
        return alpha
    
    
    def alpha_135(self):
        #### SMA(DELAY(CLOSE / DELAY(CLOSE, 20), 1), 20, 1)
        data1 = delay(self.close / delay(self.close, 20), 1)
        alpha = data1.ewm(alpha=1/20, adjust=False).mean()
        return alpha 
    
    def alpha_136(self):
        #### ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
        data1 = -1 * rank(delta(self.returns, 3))
        data2 = correlation(self.open, self.volume, 10)
        alpha = (data1 * data2)
        return alpha
    
    def alpha_137(self):
        cond3 = abs(self.low - delay(self.close, 1)) > abs(self.high - delay(self.low, 1))
        cond4 = abs(self.low - delay(self.close, 1)) > abs(self.high - delay(self.close, 1))
        part2 = (abs(self.high - delay(self.low, 1)) + abs(delay(self.close, 1) - delay(self.open, 1))) / 4 
        part2[cond3 & cond4] = abs(self.low - delay(self.close, 1)) + abs(self.high - delay(self.close, 1)) / 2 + abs(delay(self.close, 1) - delay(self.open, 1)) / 4
        cond1 = self.close - delay(self.close, 1) + (self.close - self.open) / 2 + delay(self.close, 1) - delay(self.open, 1) / abs(self.high - delay(self.close, 1)) > abs(self.low - delay(self.close, 1))
        cond2 = abs(self.high - delay(self.close, 1)) > abs(self.high - delay(self.low, 1))
        part1 = part2.copy() 
        part1[cond1 & cond2] = abs(self.high - delay(self.close, 1)) + abs(self.low - delay(self.close, 1)) / 2 + abs(delay(self.close, 1) - delay(self.open, 1)) / 4
        return 16 * part1 * np.maximum(abs(self.high - delay(self.close, 1)), abs(self.low - delay(self.close, 1)))
    
    
    def alpha_138(self):
        #### ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
        #data1 = (self.low * 0.7 + self.avg_price * 0.3).diff(3)
        #decay_weights = np.arange(1,21,1)[::-1]    # 倒序数组
        #decay_weights = decay_weights / decay_weights.sum()
        #rank1 = data1.iloc[-20:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)
        #data1 = self.low.rolling(8).apply(self.func_rank)
        #data2 = self.volume.rolling(60).mean().rolling(17).apply(self.func_rank)
        #data3 = pd.rolling_corr(data1, data2, window=5).rolling(19).apply(self.func_rank)
        #rank2 = data3.rolling(16).apply(self.func_decaylinear).iloc[-7:,:].rank(axis=0, pct=True).iloc[-1,:]
        #alpha = (rank2 - rank1).dropna()
        data1 = rank(func_decay_linear(delta(self.low * 0.7 + self.vwap * 0.3, 3), 20))
        data2 = ts_rank(func_decay_linear(ts_rank(correlation(ts_rank(self.low, 8), ts_rank(sma(self.volume, 60), 17), 5), 19), 16), 7)
        return data1 - data2
    
    def alpha_139(self):
        #### (-1 * CORR(OPEN, VOLUME, 10))
        #alpha = - self.open_price.iloc[-10:,:].corrwith(self.volume.iloc[-10:,:]).dropna()
        return -1 * correlation(self.open, self.volume, 10)
    
    def alpha_140(self):
        #### MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME, 60), 20), 8), 7), 3))
        #data1 = self.open_price.rank(axis=1, pct=True) + self.low.rank(axis=1, pct=True) \
        #        - self.high.rank(axis=1, pct=True) - self.close.rank(axis=1, pct=True)
        #rank1 = data1.iloc[-8:,:].apply(self.func_decaylinear).rank(pct=True)

        #data1 = self.close.rolling(8).apply(self.func_rank)
        #data2 = self.volume.rolling(60).mean().rolling(20).apply(self.func_rank)
        #data3 = pd.rolling_corr(data1, data2, window=8)
        #data3 = data3.rolling(7).apply(self.func_decaylinear)
        #rank2 = data3.iloc[-3:,:].rank(axis=0, pct=True).iloc[-1,:]
        #
        #'''
        #alpha = min(rank1, rank2)   NaN如何比较？
        #'''    
        data1 = rank(func_decay_linear(rank(self.open) + rank(self.low) - (rank(self.high) + rank(self.close)), 8))
        data2 = ts_rank(func_decay_linear(correlation(ts_rank(self.close, 8), ts_rank(sma(self.volume, 60), 20), 8), 7), 3)
        return np.minimum(data1, data2)
    
    def alpha_141(self):
        #### (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
        #df1 = self.high.rank(axis=1, pct=True)
        #df2 = self.volume.rolling(15).mean().rank(axis=1, pct=True)
        #alpha = -df1.iloc[-9:,:].corrwith(df2.iloc[-9:,:]).rank(pct=True)
        #alpha=alpha.dropna()
        return -1 * rank(correlation(rank(self.high), rank(sma(self.volume, 15)), 9))
    
    def alpha_142(self):
        #### (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
        #rank1 = self.close.iloc[-10:,:].rank(axis=0, pct=True).iloc[-1,:].rank(pct=True)
        #rank2 = self.close.diff(1).diff(1).iloc[-1,:].rank(pct=True)
        #rank3 = (self.volume / self.volume.rolling(20).mean()).iloc[-5:,:].rank(axis=0, pct=True).iloc[-1,:].rank(pct=True)

        #alpha = -(rank1 * rank2 * rank3).dropna()
        #alpha=alpha.dropna()
        return -1 * (rank(ts_rank(self.close, 10))) * (rank(delta(delta(self.close, 1), 1))) * (rank(ts_rank(self.volume / sma(self.volume, 20), 5)))
    
    def alpha_143(self):
        #### CLOSE > DELAY(CLOSE, 1)?(CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1) * SELF : SELF

        return 0
    
    def alpha_144(self):
        #### SUMIF(ABS(CLOSE/DELAY(CLOSE, 1) - 1)/AMOUNT, 20, CLOSE < DELAY(CLOSE, 1))/COUNT(CLOSE < DELAY(CLOSE, 1), 20)
        ##       
        cond = self.close < delay(self.close, 1)
        data = self.close.copy()
        data[cond] = 1
        data[~cond] = 0
        sumif = ts_sum((abs(self.close / delay(self.close, 1) - 1) / self.amount) * data, 20)
        count = ts_sum(data, 20)
        alpha = (sumif / count)
        return alpha
    
    
    def alpha_145(self):
        #### (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
        ##       
        #alpha = (self.volume.iloc[-9:,:].mean() - self.volume.iloc[-26:,:].mean()) / self.volume.iloc[-12:,:].mean() * 100
        #alpha=alpha.dropna()
        return (sma(self.volume, 9) - sma(self.volume, 26)) / sma(self.volume, 12) * 100
    
    
    def alpha_146(self):
        #temp1 = (self.close - delay(self.close, 1)) / delay(self.close, 1)
        #temp2 = temp1.ewm(alpha=2/61, adjust=True).mean()
	#alpha = sma(temp1 - temp2, 20) * ()
        return 0
    
    
    def alpha_147(self):
         
        return func_regbert_slope(sma(self.close, 12), 12) 
    
    
    def alpha_148(self):
        #### ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
        ##        
        df1 = ts_sum(sma(self.volume, 60), 9)
        rank1 = rank(correlation(self.open, df1, 6))
        rank2 = rank(self.open - ts_min(self.open, 14))
        alpha = -1 * (rank1 < rank2)
        return alpha
    
    
    def alpha_149(self):
        #cond = self.benchmark_close < delay(self.benchmark_self, 1) 
	#data1 = self.close.copy()
	#data1[cond] = self.close / delay(self.close, 1)- 1
	#data1[~cond] = 0
	#data2 = self.close.copy()
	#data2[cond] = self.benchmark_close / delay(self.benchmark_close, 1)- 1
	#data2[~cond] = 0
        #return func_regbert_slope(data1)
        return 0 
    
    def alpha_150(self):
        #### (CLOSE + HIGH + LOW)/3 * VOLUME
        ##       
        alpha = ((self.close + self.high + self.low) / 3 * self.volume)
        return alpha
    
    
    def alpha_151(self):
         
        return (self.close - delay(self.close, 20)).ewm(alpha=1/20, adjust=False).mean()
    
    
    ######################## alpha_152 #######################
    #       
    def alpha_152(self):
        # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1) #
        data1 = sma(delay((delay(self.close / delay(self.close, 9))).ewm(alpha=1/9, adjust=False).mean(), 1), 12)
        data2 = sma(delay((self.close / delay(self.close, 9)).ewm(alpha=1/9, adjust=False).mean(), 1), 26)
        return (data1 - data2).ewm(alpha=1/9, adjust=False).mean()
    
    
    ######################## alpha_153 #######################
    #       
    def alpha_153(self):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4 #
        alpha = ((sma(self.close, 3) + sma(self.close, 6) + sma(self.close, 12) + sma(self.close,24)) /4)
        return alpha
    
    
    ######################## alpha_154 #######################
    #       
    def alpha_154(self):
        # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18))) #
        #alpha = (self.avg_price-pd.rolling_min(self.avg_price, 16)).iloc[-1,:]<self.avg_price.iloc[-18:,:].corrwith((pd.rolling_mean(self.volume, 180)).iloc[-18:,:])
        #alpha=alpha.dropna()
        return (self.vwap - ts_min(self.vwap, 16)) < (correlation(self.vwap, sma(self.volume, 180), 18))
    
    
    ######################## alpha_155 #######################
    #       
    def alpha_155(self):
        # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2) #
        sma1 = (self.volume).ewm(alpha=2/13, adjust=False).mean()
        sma2 = (self.volume).ewm(alpha=2/27, adjust=False).mean()
        sma3 = (sma1-sma2).ewm(alpha=2/10, adjust=False).mean()
        return sma1 - sma2 - sma3
    
    ######################## alpha_156 #######################
    def alpha_156(self):
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1 #
        #rank1 = (pd.rolling_apply(self.avg_price-self.avg_price.shift(5), 3, self.func_decaylinear)).rank(axis=1, pct=True)
        #rank2 = pd.rolling_apply(-((self.open_price*0.15+self.low*0.85)-(self.open_price*0.15+self.low*0.85).shift(2))/(self.open_price*0.15+self.low*0.85), 3, self.func_decaylinear).rank(axis=1, pct=True)
        #max_cond = rank1 > rank2
        #result = rank2
        #result[max_cond] = rank1[max_cond]
        #alpha = (-result).iloc[-1,:]
        #alpha=alpha.dropna()
        data1 = rank(func_decay_linear(delta(self.vwap, 5), 3))
        data2 = rank(func_decay_linear((delta(self.open * 0.15 + self.low * 0.85, 2)) / (self.open * 0.15 + self.low * 0.85) * -1, 3))
        return np.maximum(data1, data2) * -1
    
    ######################## alpha_157 #######################
    def alpha_157(self):
        # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5)) #
        #rank1 = (-((self.close-1)-(self.close-1).shift(5)).rank(axis=1, pct=True)).rank(axis=1, pct=True).rank(axis=1, pct=True)
        #min1 = rank1.rolling(2).min()
        #log1 = np.log(min1)
        #rank2 = log1.rank(axis=1, pct=True).rank(axis=1, pct=True)
        #cond_min = rank2 > 5
        #rank2[cond_min] = 5
        #tsrank1 = pd.rolling_apply((-((self.close/self.prev_close)-1)).shift(6), 5, self.func_rank)
        #alpha = (rank2+tsrank1).iloc[-1,:]
        #alpha=alpha.dropna()
        part1 = np.minimum(product(rank(rank(log(ts_sum(ts_min(rank(rank(-rank(delta(self.close - 1, 5)))), 2), 1)))), 1), 5) 
        part2 = ts_rank(delay(-self.returns, 6), 5)
        return part1 + part2
    
    ######################## alpha_158 #######################
    #       
    def alpha_158(self):
        # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE #
        #alpha = (((self.high-pd.ewma(self.close, span=14, adjust=False))-(self.low-pd.ewma(self.close, span=14, adjust=False)))/self.close).iloc[-1,:]
        part = (self.close).ewm(alpha=2/15, adjust=False).mean()
        return (self.high - part) - (self.low - part) / self.close
    
    
    def alpha_159(self):
        #########((close-sum(min(low,delay(close,1)),6))/sum(max(high,delay(close,1))-min(low,delay(close,1)),6)*12*24+(close-sum(min(low,delay(close,1)),12))/sum(max(high,delay(close,1))-min(low,delay(close,1)),12)*6*24+(close-sum(min(low,delay(close,1)),24))/sum(max(high,delay(close,1))-min(low,delay(close,1)),24)*6*24)*100/(6*12+6*24+12*24)
        ###################      
        data1 = self.low.copy()
        data2 = delay(self.close, 1)
        data3 = self.high.copy()
        min_data = np.minimum(data1, data2)
        max_data = np.maximum(data3, data2)
        x = ((self.close - ts_sum(min_data,6)) / ts_sum((max_data - min_data), 6)) * 12 * 24
        y = ((self.close - ts_sum(min_data,12)) / ts_sum((max_data - min_data), 12)) * 6 * 24
        z = ((self.close - ts_sum(min_data,24)) / ts_sum((max_data - min_data), 24)) * 6 * 24
        alpha = (x + y + z) * (100 / (6 * 12 + 12 * 24 + 6 * 24))
        return alpha
    
    
    def alpha_160(self):
        ################      
        ############sma((close<=delay(close,1)?std(close,20):0),20,1)
        data1 = stddev(self.close, 20)
        cond = self.close > delay(self.close, 1)
        data1[cond] = 0
        return data1.ewm(alpha=1/20, adjust=False).mean()
    
    
    def alpha_161(self):
        ###########mean((max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),12)
        ################      
        data1 = (self.high - self.low)
        data2 = abs(delay(self.close, 1) - self.high)
        data3 = abs(delay(self.close, 1) - self.low)
        alpha = sma(np.maximum(np.maximum(data1, data2), data3), 12)
        return alpha 
    
    def alpha_162(self):
        ###############(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100,12))/(max(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12)-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12))
        #################      
        #算出公式核心部分X
        data1 = self.close - delay(self.close, 1)
        max_data = np.maximum(data1, 0)
        part1 = (max_data).ewm(alpha=1/12, adjust=False).mean() / (abs(data1)).ewm(alpha=1/12, adjust=False).mean() * 100
        part2 = np.minimum(max_data.ewm(alpha=1/12, adjust=False).mean() / abs(data1).ewm(alpha=1/12, adjust=False).mean() * 100, 12) 
        part3 = np.maximum(max_data.ewm(alpha=1/12, adjust=False).mean() / abs(data1).ewm(alpha=1/12, adjust=False).mean() * 100, 12)
        return (part1 - part2) / (part2 - part3)
    
    def alpha_163(self):
        ################      
        #######rank(((((-1*ret)*,mean(volume,20))*vwap)*(high-close)))
        #data1=(-1)*(self.close/self.close.shift()-1)*pd.rolling_mean(self.volume,20)*self.avg_price*(self.high-self.close)
        #data2=(data1.rank(axis=1,pct=True)).iloc[-1,:]
        #alpha=data2
        #alpha=alpha.dropna()
        return rank((-1 * self.returns * sma(self.volume, 20) * self.vwap) * (self.high - self.close))
    
    def alpha_164(self):
        ################      
        ############sma((((close>delay(close,1))?1/(close-delay(close,1)):1)-min(((close>delay(close,1))?1/(close/delay(close,1)):1),12))/(high-low)*100,13,2)
        cond = self.close <= delay(self.close, 1)
        data1 = 1 / (self.close - delay(self.close, 1))
        data1[cond] = 1
        data2 = np.minimum(data1, 12)
        data3 = (data1 - data2) / ((self.high - self.low) * 100)
        return data3.ewm(alpha=2/13, adjust=False).mean()
   
    
    def alpha_165(self):
         
        return 0  

    
    def alpha_166(self):
        return 0    
    
    
    def alpha_167(self):
        ##      
        ####sum(((close-delay(close,1)>0)?(close-delay(close,1)):0),12)####
        data1 = self.close - delay(self.close, 1)
        cond = (data1 <= 0)
        alpha = data1
        alpha[cond] = 0
        return ts_sum(alpha, 12)
    
    
    def alpha_168(self):
        ##      
        #####-1*volume/mean(volume,20)####
        alpha = (-1 * self.volume) / sma(self.volume, 20)
        return alpha
    
    
    def alpha_169(self):
        ##      
        ###sma(mean(delay(sma(close-delay(close,1),9,1),1),12)-mean(delay(sma(close-delay(close,1),9,1),1),26),10,1)#####
        data1 = self.close - delay(self.close, 1)
        data2 = delay(data1.ewm(alpha=1/9, adjust=False).mean(), 1)
        data3 = sma(data2, 12) 
        data4 = sma(data2, 26) 
        alpha = (data3 - data4).ewm(alpha=1/10, adjust=False).mean()
        return alpha  
    
    
    def alpha_170(self):
        ##      
        #####((((rank((1/close))*volume)/mean(volume,20))*((high*rank((high-close)))/(sum(high,5)/5)))-rank((vwap-delay(vwap,5))))####
        data1 = rank(1 / self.close) * self.volume / sma(self.volume, 20)
        data2 = self.high * rank(self.high - self.close) / (ts_sum(self.high, 5) / 5)
        data3 = rank(self.vwap - delay(self.vwap, 5))
        return data1 * data2 - data3
    
    
    def alpha_171(self):
        ##      
        ####(((low-close)*open^5)*-1)/((close-high)*close^5)#####
        data1 = -1 * (self.low - self.close) * (self.open ** 5)
        data2 = (self.close - self.high) * (self.close ** 5)
        alpha = (data1 / data2)
        return alpha
    
    
    #       
    def alpha_172(self):
        # MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6) #
        hd = self.high - delay(self.high, 1)
        ld = delay(self.low, 1) - self.low
        temp1 = self.high - self.low
        temp2 = abs(self.high - delay(self.close, 1))
        cond1 = temp1 > temp2
        temp2[cond1] = temp1[cond1]
        temp3 = abs(self.low - delay(self.close, 1))
        cond2 = temp2>temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3   # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = ts_sum(tr, 14)
        cond3 = ld>0
        cond4 = ld>hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = ts_sum(data1, 14) * 100 / sum_tr14
        cond5 = hd>0
        cond6 = hd>ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = ts_sum(data2, 14) * 100 / sum_tr14
        alpha = sma(abs(sum1 - sum2) / (sum1 + sum2) * 100, 6)
        return alpha
    
    
    def alpha_173(self):
        ##      
        ####3*sma(close,13,2)-2*sma(sma(close,13,2),13,2)+sma(sma(sma(log(close),13,2),13,2),13,2)#####
        data1 = self.close.ewm(alpha=2/13, adjust=False).mean()
        data2 = data1.ewm(alpha=2/13, adjust=False).mean()
        close_log = np.log(self.close)
        data3 = close_log.ewm(alpha=2/13, adjust=False).mean()
        data4 = data3.ewm(alpha=2/13, adjust=False).mean()
        data5 = data4.ewm(alpha=2/13, adjust=False).mean()
        alpha = (3 * data1 - 2 * data2 + data5)
        return alpha
    
    
    def alpha_174(self):
        ##      
        ####sma((close>delay(close,1)?std(close,20):0),20,1)#####
        cond = self.close <= delay(self.close, 1)
        data2 = stddev(self.close, 20)
        data2[cond] = 0
        return data2.ewm(alpha=1/20, adjust=False).mean()
    
    
    def alpha_175(self):
        ##      
        #####mean(max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),6)####
        data1 = self.high - self.low
        data2 = abs(delay(self.close, 1) - self.high)
        data3 = abs(delay(self.close, 1) - self.low)
        data4 = np.maximum(np.maximum(data1, data2), data3)
        alpha = sma(data4, 6)
        return alpha
    
    
    def alpha_176(self):
        ##      
        ######### #########corr(rank((close-tsmin(low,12))/(tsmax(high,12)-tsmin(low,12))),rank(volume),6)#############
        data1 = rank((self.close - ts_min(self.low, 12)) / (ts_max(self.high, 12) - ts_min(self.low, 12)))
        data2 = rank(self.volume)
        alpha = correlation(data1, data2, 6)
        return alpha
    
    
    ################## alpha_177 ####################
    #       
    def alpha_177(self):
        ##### ((20-HIGHDAY(HIGH,20))/20)*100 #####
        #alpha = (20 - self.high.iloc[-20:,:].apply(self.func_highday))/20*100
        #alpha=alpha.dropna()
        return (20 - func_high_day(self.high, 20)) / 20 * 100
    
    
    def alpha_178(self):
        ##### (close-delay(close,1))/delay(close,1)*volume ####
        ##       
        alpha = ((self.close - delay(self.close, 1)) / delay(self.close, 1) *self.volume)
        return alpha
    
    
    def alpha_179(self):
        #####（rank(corr(vwap,volume,4))*rank(corr(rank(low),rank(mean(volume,50)),12))####
        ##       
        rank1 = rank(correlation(self.vwap, self.volume, 4))
        rank2 = rank(correlation(rank(self.low), rank(sma(self.volume, 50)), 12))
        alpha = rank1 * rank2
        return alpha 
    
    
    ##################### alpha_180 #######################
    #       
    def alpha_180(self):
        ##### ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME))) #####
        ma = sma(self.volume, 20)
        cond = (ma < self.volume)
        alpha = -self.volume
        alpha[cond] = (-ts_rank(abs(delta(self.close, 7)), 60)) * sign(delta(self.close, 7))
        return alpha

    def alpha_181(self):
        data1 = (self.close / delay(self.close, 1) - 1) - sma(self.close / delay(self.close, 1) - 1, 20)
        data2 = self.benchmark_close - sma(self.benchmark_close, 20) 
        alpha = ts_sum(data1 - data2 ** 2, 20) / ts_sum(data2 ** 3, 20)
        return alpha
    
    
    ######################## alpha_182 #######################
    def alpha_182(self):
        ##### COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20 #####
        cond1 = (self.close > self.open)
        cond2 = (self.benchmark_open > self.benchmark_close)
        cond3 = (self.close < self.open)
        cond4 = (self.benchmark_open < self.benchmark_close)
        data = self.close.copy()
        data[(cond1 & cond2) | (cond3 & cond4)] = 1
        data[~((cond1 & cond2) | (cond3 & cond4))] = 0
        return ts_sum(data, 20) / 20
    
    
    def alpha_183(self):
      
        return 0
    
    
    def alpha_184(self):
        #####(rank(corr(delay((open-close),1),close,200))+rank((open-close))) ####
        ##       
        data1 = delay(self.open, 1) - delay(self.close, 1)
        data2 = rank(self.open - self.close)
        alpha = rank(correlation(data1, self.close, 200)) + data2
        return alpha
 
    
    def alpha_185(self):
        ##### RANK((-1 * ((1 - (OPEN / CLOSE))^2))) ####
        alpha = rank(-(1-self.open / self.close) ** 2)
        return alpha
    
    
    #       
    def alpha_186(self):
        # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2 #
        hd = self.high - delay(self.high, 1)
        ld = delay(self.low, 1) - self.low
        temp1 = self.high - self.low
        temp2 = abs(self.high - delay(self.close, 1))
        cond1 = temp1>temp2
        temp2[cond1] = temp1[cond1]
        temp3 = abs(self.low - delay(self.close, 1))
        cond2 = temp2>temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3   # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = ts_sum(tr, 14)
        cond3 = ld>0
        cond4 = ld>hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = ts_sum(data1, 14) * 100 / sum_tr14
        cond5 = hd>0
        cond6 = hd>ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = ts_sum(data2, 14) * 100 / sum_tr14
        mean1 = sma(abs(sum1-sum2) / (sum1+sum2) * 100, 6)
        alpha = ((mean1 + delay(mean1, 6)) / 2)
        return alpha
    
    
    def alpha_187(self):
        ##### SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20) ####
        cond = (self.open <= delay(self.open, 1))
        data1 = self.high - self.low                        # HIGH-LOW
        data2 = self.open - delay(self.open, 1)   # OPEN-DELAY(OPEN,1)
        alpha = np.maximum(data1, data2)
        alpha[cond] = 0
        return ts_sum(alpha, 20)

    
    def alpha_188(self):
        ##### ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100 #####
        data1 = (self.high - self.low).ewm(alpha=2/11, adjust=False).mean()   
        alpha = ((self.high - self.low - data1) / data1 * 100)
        return alpha
    
    
    def alpha_189(self):
        ##### mean(abs(close-mean(close,6),6)) ####
        ma6 = sma(self.close, 6)  
        alpha = sma(abs(self.close - ma6), 6)
        return alpha
    
    
    def alpha_190(self):
        ##### LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE)
        ##### -1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((
        ##### COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE
        ##### /DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1)))) ####
        
        return 0
    
    
    def alpha_191(self):
        ##### (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE ####
        #      /chencheng
        #volume_avg = pd.rolling_mean(self.volume, window=20)
        #corr = volume_avg.iloc[-5:,:].corrwith(self.low.iloc[-5:,:])    
        #alpha = corr + (self.high.iloc[-1,:] + self.low.iloc[-1,:])/2 - self.close.iloc[-1,:]
        #alpha=alpha.dropna()
        return correlation(sma(self.volume, 20), self.low, 5) + (self.high + self.low) / 2 - self.close

if __name__=='__main__':
    df = pd.read_csv('../data/0721_300676_part.csv')
    print(df.head())
    df_benchmark = pd.read_excel('../data/0721_CSI300.xlsx')
    print(df_benchmark.head())
    stock = GTJA_191(df, df_benchmark)
    df['alpha001'] = stock.alpha_001()
    df['alpha002'] = stock.alpha_002()
    df['alpha003'] = stock.alpha_003()
    df['alpha004'] = stock.alpha_004()
    df['alpha005'] = stock.alpha_005()
    df['alpha006'] = stock.alpha_006()
    df['alpha007'] = stock.alpha_007()
    df['alpha008'] = stock.alpha_008()
    df['alpha009'] = stock.alpha_009()
    df['alpha010'] = stock.alpha_010()
    df['alpha011'] = stock.alpha_011()
    df['alpha012'] = stock.alpha_012()
    df['alpha013'] = stock.alpha_013()
    df['alpha014'] = stock.alpha_014()
    df['alpha015'] = stock.alpha_015()
    df['alpha016'] = stock.alpha_016()
    df['alpha017'] = stock.alpha_017()
    df['alpha018'] = stock.alpha_018()
    df['alpha019'] = stock.alpha_019()
    df['alpha020'] = stock.alpha_020()
    df['alpha021'] = stock.alpha_021()
    df['alpha022'] = stock.alpha_022()
    df['alpha023'] = stock.alpha_023()
    df['alpha024'] = stock.alpha_024()
    df['alpha025'] = stock.alpha_025()
    df['alpha026'] = stock.alpha_026()
    df['alpha027'] = stock.alpha_027()
    df['alpha028'] = stock.alpha_028()
    df['alpha029'] = stock.alpha_029()
    #df['alpha030'] = stock.alpha_030()
    df['alpha031'] = stock.alpha_031()
    df['alpha032'] = stock.alpha_032()
    df['alpha033'] = stock.alpha_033()
    df['alpha034'] = stock.alpha_034()
    df['alpha035'] = stock.alpha_035()
    df['alpha036'] = stock.alpha_036()
    df['alpha037'] = stock.alpha_037()
    df['alpha038'] = stock.alpha_038()
    df['alpha039'] = stock.alpha_039()
    df['alpha040'] = stock.alpha_040()
    df['alpha041'] = stock.alpha_041()
    df['alpha042'] = stock.alpha_042()
    df['alpha043'] = stock.alpha_043()
    df['alpha044'] = stock.alpha_044()
    df['alpha045'] = stock.alpha_045()
    df['alpha046'] = stock.alpha_046()
    df['alpha047'] = stock.alpha_047()
    df['alpha048'] = stock.alpha_048()
    df['alpha049'] = stock.alpha_049()
    df['alpha050'] = stock.alpha_050()
    df['alpha051'] = stock.alpha_051()
    df['alpha052'] = stock.alpha_052()
    df['alpha053'] = stock.alpha_053()
    df['alpha054'] = stock.alpha_054()
    df['alpha055'] = stock.alpha_055()
    df['alpha056'] = stock.alpha_056()
    df['alpha057'] = stock.alpha_057()
    df['alpha058'] = stock.alpha_058()
    df['alpha059'] = stock.alpha_059()
    df['alpha060'] = stock.alpha_060()
    df['alpha061'] = stock.alpha_061()
    df['alpha062'] = stock.alpha_062()
    df['alpha063'] = stock.alpha_063()
    df['alpha064'] = stock.alpha_064()
    df['alpha065'] = stock.alpha_065()
    df['alpha066'] = stock.alpha_066()
    df['alpha067'] = stock.alpha_067()
    df['alpha068'] = stock.alpha_068()
    df['alpha069'] = stock.alpha_069()
    df['alpha070'] = stock.alpha_070()
    df['alpha071'] = stock.alpha_071()
    df['alpha072'] = stock.alpha_072()
    df['alpha073'] = stock.alpha_073()
    df['alpha074'] = stock.alpha_074()
    df['alpha075'] = stock.alpha_075()
    df['alpha076'] = stock.alpha_076()
    df['alpha077'] = stock.alpha_077()
    df['alpha078'] = stock.alpha_078()
    df['alpha079'] = stock.alpha_079()
    df['alpha080'] = stock.alpha_080()
    df['alpha081'] = stock.alpha_081()
    df['alpha082'] = stock.alpha_082()
    df['alpha083'] = stock.alpha_083()
    df['alpha084'] = stock.alpha_084()
    df['alpha085'] = stock.alpha_085()
    df['alpha086'] = stock.alpha_086()
    df['alpha087'] = stock.alpha_087()
    df['alpha088'] = stock.alpha_088()
    df['alpha089'] = stock.alpha_089()
    df['alpha090'] = stock.alpha_090()
    df['alpha091'] = stock.alpha_091()
    df['alpha092'] = stock.alpha_092()
    df['alpha093'] = stock.alpha_093()
    df['alpha094'] = stock.alpha_094()
    df['alpha095'] = stock.alpha_095()
    df['alpha096'] = stock.alpha_096()
    df['alpha097'] = stock.alpha_097()
    df['alpha098'] = stock.alpha_098()
    df['alpha099'] = stock.alpha_099()
    df['alpha100'] = stock.alpha_100()
    df['alpha101'] = stock.alpha_101()
    df['alpha102'] = stock.alpha_102()
    df['alpha103'] = stock.alpha_103()
    df['alpha104'] = stock.alpha_104()
    df['alpha105'] = stock.alpha_105()
    df['alpha106'] = stock.alpha_106()
    df['alpha107'] = stock.alpha_107()
    df['alpha108'] = stock.alpha_108()
    df['alpha109'] = stock.alpha_109()
    df['alpha110'] = stock.alpha_110()
    df['alpha111'] = stock.alpha_111()
    df['alpha112'] = stock.alpha_112()
    df['alpha113'] = stock.alpha_113()
    df['alpha114'] = stock.alpha_114()
    df['alpha115'] = stock.alpha_115()
    df['alpha116'] = stock.alpha_116()
    df['alpha117'] = stock.alpha_117()
    df['alpha118'] = stock.alpha_118()
    df['alpha119'] = stock.alpha_119()
    df['alpha120'] = stock.alpha_120()
    df['alpha121'] = stock.alpha_121()
    df['alpha122'] = stock.alpha_122()
    df['alpha123'] = stock.alpha_123()
    df['alpha124'] = stock.alpha_124()
    df['alpha125'] = stock.alpha_125()
    df['alpha126'] = stock.alpha_126()
    #df['alpha127'] = stock.alpha_127()
    df['alpha128'] = stock.alpha_128()
    df['alpha129'] = stock.alpha_129()
    df['alpha130'] = stock.alpha_130()
    df['alpha131'] = stock.alpha_131()
    df['alpha132'] = stock.alpha_132()
    df['alpha133'] = stock.alpha_133()
    df['alpha134'] = stock.alpha_134()
    df['alpha135'] = stock.alpha_135()
    df['alpha136'] = stock.alpha_136()
    df['alpha137'] = stock.alpha_137()
    df['alpha138'] = stock.alpha_138()
    df['alpha139'] = stock.alpha_139()
    df['alpha140'] = stock.alpha_140()
    df['alpha141'] = stock.alpha_141()
    df['alpha142'] = stock.alpha_142()
    #df['alpha143'] = stock.alpha_143()
    df['alpha144'] = stock.alpha_144()
    df['alpha145'] = stock.alpha_145()
    #df['alpha146'] = stock.alpha_146()
    df['alpha147'] = stock.alpha_147()
    df['alpha148'] = stock.alpha_148()
    #df['alpha149'] = stock.alpha_149()
    df['alpha150'] = stock.alpha_150()
    df['alpha151'] = stock.alpha_151()
    df['alpha152'] = stock.alpha_152()
    df['alpha153'] = stock.alpha_153()
    df['alpha154'] = stock.alpha_154()
    df['alpha155'] = stock.alpha_155()
    df['alpha156'] = stock.alpha_156()
    df['alpha157'] = stock.alpha_157()
    df['alpha158'] = stock.alpha_158()
    df['alpha159'] = stock.alpha_159()
    df['alpha160'] = stock.alpha_160()
    df['alpha161'] = stock.alpha_161()
    df['alpha162'] = stock.alpha_162()
    df['alpha163'] = stock.alpha_163()
    df['alpha164'] = stock.alpha_164()
    #df['alpha165'] = stock.alpha_165()
    #df['alpha166'] = stock.alpha_166()
    df['alpha167'] = stock.alpha_167()
    df['alpha168'] = stock.alpha_168()
    df['alpha169'] = stock.alpha_169()
    df['alpha170'] = stock.alpha_170()
    df['alpha171'] = stock.alpha_171()
    df['alpha172'] = stock.alpha_172()
    df['alpha173'] = stock.alpha_173()
    df['alpha174'] = stock.alpha_174()
    df['alpha175'] = stock.alpha_175()
    df['alpha176'] = stock.alpha_176()
    df['alpha177'] = stock.alpha_177()
    df['alpha178'] = stock.alpha_178()
    df['alpha179'] = stock.alpha_179()
    df['alpha180'] = stock.alpha_180()
    #df['alpha181'] = stock.alpha_181()
    df['alpha182'] = stock.alpha_182()
    #df['alpha183'] = stock.alpha_183()
    df['alpha184'] = stock.alpha_184()
    df['alpha185'] = stock.alpha_185()
    df['alpha186'] = stock.alpha_186()
    df['alpha187'] = stock.alpha_187()
    df['alpha188'] = stock.alpha_188()
    df['alpha189'] = stock.alpha_189()
    #df['alpha190'] = stock.alpha_190()
    df['alpha191'] = stock.alpha_191()
    print("Alpha191 computed sucessful!")
    df.to_excel('../data/alpha191_0721_300676.xls', index=None)
