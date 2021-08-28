import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config


def load_dataset(symbol):
    ohlcv = pd.read_csv(f'data/{symbol}-5m-data.csv')
    ohlcv['rank_'] = ohlcv.groupby('timestamp')['volume'].rank(method='first', ascending=True)
    ohlcv = ohlcv[ohlcv.rank_ == 1]
    ohlcv['timestamp'] = pd.to_datetime(ohlcv['timestamp'])
    ohlcv = ohlcv.drop(columns=['rank_'])
    return ohlcv


def data_split(df, start=None, end=None):
    if start is not None:
        data = df[(df.timestamp >= start) & (df.timestamp < end)]
    else:
        data = df[(df.timestamp < end)]

    data = data.sort_values(['timestamp', 'symbol'], ignore_index=True)
    data.index = data.timestamp.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df

from talib import abstract
import itertools

def add_talib_feature(ohlcv):
    # BBANDS
    for matype in range(9):
        for timeperiod in list(range(5, 21, 5)):
            upperband, middleband, lowerband = abstract.BBANDS(
                ohlcv,
                timeperiod=timeperiod,
                matype=matype
            )
            ohlcv['BBANDS_{:02d}{:d}_upperband'.format(timeperiod, matype)] = upperband
            ohlcv['BBANDS_{:02d}{:d}_middleband'.format(timeperiod, matype)] = middleband
            ohlcv['BBANDS_{:02d}{:d}_lowerband'.format(timeperiod, matype)] = lowerband

    # STOCH
    para_combs = list(itertools.permutations(list(range(5, 46, 10)), 3))
    for matype in range(9):
        for para_comb in para_combs:
            slowk, slowd = abstract.STOCH(
                ohlcv,
                fastk_period=para_comb[0],
                slowk_period=para_comb[1],
                slowd_period=para_comb[2],
                slowk_matype=matype,
                slowd_matype=matype
            )
            ohlcv['STOCH_{:02d}{:02d}{:02d}{:d}{:d}_slowk'.format(para_comb[0], para_comb[1], para_comb[2], matype, matype)] = slowk
            ohlcv['STOCH_{:02d}{:02d}{:02d}{:d}{:d}_slowd'.format(para_comb[0], para_comb[1], para_comb[2], matype, matype)] = slowd

    # MACD
    para_combs = list(itertools.permutations(list(range(5, 46, 10)), 3))
    for para_comb in para_combs:
        macd, macdsignal, macdhist = abstract.MACD(
            ohlcv,
            fastperiod=para_comb[0],
            slowperiod=para_comb[1],
            signalperiod=para_comb[2]
        )
        ohlcv['MACD_{:02d}{:02d}{:02d}_macd'.format(para_comb[0], para_comb[1], para_comb[2])] = macd
        ohlcv['MACD_{:02d}{:02d}{:02d}_macdsignal'.format(para_comb[0], para_comb[1], para_comb[2])] = macdsignal
        ohlcv['MACD_{:02d}{:02d}{:02d}_macdhist'.format(para_comb[0], para_comb[1], para_comb[2])] = macdhist

    # RSI
    for timeperiod in list(range(5, 41, 3)):
        ohlcv['RSI_{:02d}'.format(timeperiod)] = abstract.RSI(
            ohlcv,
            timeperiod=timeperiod
        )

    # CCI
    for timeperiod in list(range(5, 41, 3)):
        ohlcv['CCI_{:02d}'.format(timeperiod)] = abstract.CCI(
            ohlcv,
            timeperiod=timeperiod
        )

    # ADX
    for timeperiod in list(range(5, 41, 3)):
        ohlcv['ADX_{:02d}'.format(timeperiod)] = abstract.ADX(
            ohlcv,
            timeperiod=timeperiod
        )

    ohlcv = ohlcv.fillna(method='ffill')
    return ohlcv

def preprocess_data():
    """data preprocessing pipeline"""
    all_ohlcv = pd.DataFrame()
    for symbol in config.SYMBOLS:
        ohlcv = load_dataset(symbol)
        ohlcv = ohlcv[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']]
        ohlcv['symbol'] = symbol
        ohlcv = ohlcv.sort_values(by=['timestamp'])
        ohlcv_feature = add_talib_feature(ohlcv)
        # all_ohlcv = all_ohlcv.append(ohlcv, ignore_index=True)

    return all_ohlcv


def add_turbulence(ohlcv):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(ohlcv):
    """calculate turbulence index based on dow 30"""
    # can add other market assets

    df_price_pivot = ohlcv.pivot(index='timestamp', columns='symbol', values='close')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)


    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index










