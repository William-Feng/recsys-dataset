"""
This module is used to provide helper functions other covis files (used for the predictions file).

It also exposes the covisitation matrices for other models to use as feature engineering.
However, these are not used in the final pipeline.
"""


import pandas as pd
import numpy as np
import glob


TRAIN_PATH = "../../test/resources/train_parquet/*"
TEST_PATH = "../../test/resources/test_parquet/*"
TYPE_LABELS = {"clicks": 0, "carts": 1, "orders": 2}
TYPE_WEIGHT = {0: 1, 1: 6, 2: 3}
PREDICTIONS_OUTPUT_PATH = "../../covis-predictions.csv"
TIME_MS_24_HRS = 24 * 60 * 60
TOP_N = 20
MAX_DISK_SIZE = 2000000
CLICKS_COVIS_FILE_PREFIX = "../test/resources/covis/clicks"
CARTS_TO_ORDERS_COVIS_FILE_PREFIX = "../test/resources/covis/carts_to_orders"
BUY_TO_BUY_COVIS_FILE_PREFIX = "../test/resources/covis/buy_to_buy"

TS_BEGIN = 1659304800
TS_END = 1662328791


def read_file(f):
    """
    Reads the DataFrame from the cache.

    Params:
        f (str): file path to be read from the cache.

    Returns:
        pd.DataFrame: DataFrame read from the cache.
    """

    return pd.DataFrame(data_cache[f])


def read_file_to_cache(f):
    """
    Reads parquet file, performs necessary transformations and adds it to the cache.

    Params:
        f (str): file path of the parquet file.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """

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


def pqt_to_dict(dataframe):
    """
    Converts DataFrame to nested dictionary based on groupby.

    Params:
    dataframe (pandas.DataFrame): Input DataFrame with 'aid_x' and 'aid_y' columns.

    Returns:
    dict: Nested dictionary with 'aid_x' values as keys and lists of corresponding 'aid_y' values as values.
    """

    return dataframe.groupby("aid_x").aid_y.apply(list).to_dict()


def get_covis_matrices_df(NUM_SECTION=4):
    """
    Loads and concatenates co-visit matrix files.

    Loads co-visit matrix data files, concatenates them and returns a dictionary
    with key as file type and value as corresponding DataFrame.

    Params:
        NUM_SECTION (int): Number of sections to read for each type of co-visit matrix. Default is 4.

    Returns:
        dict: Dictionary with keys as file type and values as corresponding DataFrame.
    """

    clicks_covis = [pd.read_parquet(f"{CLICKS_COVIS_FILE_PREFIX}_0.pqt")]

    for k in range(1, NUM_SECTION):
        clicks_covis.append(pd.read_parquet(f"{CLICKS_COVIS_FILE_PREFIX}_{k}.pqt"))

    clicks_covis = pd.concat(clicks_covis)

    carts_to_orders_covis = [
        pd.read_parquet(f"{CARTS_TO_ORDERS_COVIS_FILE_PREFIX}_0.pqt")
    ]
    for k in range(1, NUM_SECTION):
        carts_to_orders_covis.append(
            pd.read_parquet(f"{CARTS_TO_ORDERS_COVIS_FILE_PREFIX}_{k}.pqt")
        )
    carts_to_orders_covis = pd.concat(carts_to_orders_covis)

    buy_to_buy_covis = [pd.read_parquet(f"{BUY_TO_BUY_COVIS_FILE_PREFIX}_0.pqt")]

    return {
        "clicks_covis": clicks_covis,
        "carts_to_orders_covis": carts_to_orders_covis,
        "buy_to_buy_covis": buy_to_buy_covis,
    }


def get_covis_matrices_dict(NUM_SECTION=4):
    """
    Loads and converts co-visit matrix files to dictionary.

    Loads co-visit matrix data files, converts them to dictionary
    and returns a dictionary with key as file type and value as corresponding dict.

    Params:
        NUM_SECTION (int): Number of sections to read for each type of co-visit matrix. Default is 4.

    Returns:
        dict: Dictionary with keys as file type and values as corresponding dict.
    """

    clicks_covis = pqt_to_dict(pd.read_parquet(f"{CLICKS_COVIS_FILE_PREFIX}_0.pqt"))
    for k in range(1, NUM_SECTION):
        clicks_covis.update(
            pqt_to_dict(pd.read_parquet(f"{CLICKS_COVIS_FILE_PREFIX}_{k}.pqt"))
        )

    carts_to_orders_covis = pqt_to_dict(
        pd.read_parquet(f"{CARTS_TO_ORDERS_COVIS_FILE_PREFIX}_0.pqt")
    )
    for k in range(1, NUM_SECTION):
        carts_to_orders_covis.update(
            pqt_to_dict(pd.read_parquet(f"{CARTS_TO_ORDERS_COVIS_FILE_PREFIX}_{k}.pqt"))
        )

    buy_to_buy_covis = pqt_to_dict(
        pd.read_parquet(f"{BUY_TO_BUY_COVIS_FILE_PREFIX}_0.pqt")
    )

    return {
        "clicks_covis": clicks_covis,
        "carts_to_orders_covis": carts_to_orders_covis,
        "buy_to_buy_covis": buy_to_buy_covis,
    }
