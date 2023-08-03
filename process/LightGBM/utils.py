"""
This module is used to provide helper functions for the LGBM model.
"""


import pandas as pd
import numpy as np
import glob


TRAIN_PATH = "../../test/resources/train_parquet/*"
TEST_PATH = "../../test/resources/test_parquet/*"
TYPE_LABELS = {"clicks": 0, "carts": 1, "orders": 2}
TYPE_WEIGHT = {0: 1, 1: 6, 2: 3}
PREDICTIONS_OUTPUT_PATH = "predictions.csv"

TOP_N = 20

CLICKS_COVIS_FILE_PREFIX = "clicks_covis"
CARTS_TO_ORDERS_COVIS_FILE_PREFIX = "carts_to_orders"
BUY_TO_BUY_COVIS_FILE_PREFIX = "buy_to_buy"

TS_BEGIN = 1659304800
TS_END = 1662328791


def read_file(f):
    return pd.DataFrame(data_cache[f])


def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype("int32")
    df["type"] = df["type"].map(TYPE_LABELS).astype("int8")
    return df


data_cache = {}
files = glob.glob(TRAIN_PATH)
for f in files:
    data_cache[f] = read_file_to_cache(f)

READ_CT = 5
CHUNK = int(np.ceil(len(files) / 6))
