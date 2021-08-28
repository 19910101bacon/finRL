import datetime
import os

TRAINED_MODEL_DIR = f"trained_models/{datetime.datetime.now()}"
os.makedirs(TRAINED_MODEL_DIR)
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

# TESTING_DATA_FILE = "test.csv"

SYMBOLS = ['BTCUSDT', 'BNBUSDT', 'ETHUSDT', 'LTCUSDT', 'DASHUSDT']
STOCK_DIM = len(SYMBOLS)
TEST_TIMESTAMP = '2021-02-01 00:00:00'
REBALANCE_WINDOW_DATE_NUM = 20
VALIDATION_WINDOW_DATE_NUM = 20
REBALANCE_WINDOW_TIMESTAMPE_NUM = REBALANCE_WINDOW_DATE_NUM*288
VALIDATION_WINDOW_TIMESTAMPE_NUM = VALIDATION_WINDOW_DATE_NUM*288

OHLCV_FILE_PATH = 'data/symbol5.csv'


FEATURE_LIST = []
OBSERVATION_SPACE = 1 + STOCK_DIM + STOCK_DIM + FEATURE_LIST*STOCK_DIM

#########
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000000
# total number of stocks in our portfolio
# STOCK_DIM = 5
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

# turbulence index: 90-150 reasonable threshold
REWARD_SCALING = 1e-4
