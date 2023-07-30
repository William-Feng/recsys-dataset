import pandas as pd
import gc

from utils import (
    BUY_TO_BUY_COVIS_FILE_PREFIX,
    CARTS_TO_ORDERS_COVIS_FILE_PREFIX,
    CHUNK,
    CLICKS_COVIS_FILE_PREFIX,
    READ_CT,
    TS_BEGIN,
    TS_END,
    TYPE_WEIGHT,
    files,
    read_file,
)


def clicks_covis(NUM_SECTIONS=4, TOTAL_SIZE=1.86e6 / 4):
    """
    Compute the co-visitation matrix for clicks.

    This function processes the data in parts for memory management and generates the co-visitation matrix.
    It performs the following steps:
    1. Divide the data into 'NUM_SECTIONS' parts for efficient processing.
    2. Load the data in chunks and merge them in groups of 'READ_CT'.
    3. Use the tail of each session, keeping only up to 30 events per session.
    4. Create pairs of articles based on session information and timestamps.
    5. Assign weights to each pair based on timestamps and calculate the sum of weights for each pair.
    6. Combine the results from inner and outer chunks to form the co-visitation matrix.
    7. Convert the matrix to a dictionary and save the top 40 pairs for each article to disk.
    """
    print("Creating clicks covisitation matrix")

    for PART in range(NUM_SECTIONS):
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))

            for k in range(a, b, READ_CT):
                dataframe = [read_file(files[k])]
                for i in range(1, READ_CT):
                    if k + i < b:
                        dataframe.append(read_file(files[k + i]))
                dataframe = pd.concat(dataframe, ignore_index=True, axis=0)
                dataframe = dataframe.sort_values(
                    ["session", "ts"], ascending=[True, False]
                )

                dataframe = dataframe.reset_index(drop=True)
                dataframe["n"] = dataframe.groupby("session").cumcount()
                dataframe = dataframe.loc[dataframe.n < 30].drop("n", axis=1)

                dataframe = dataframe.merge(dataframe, on="session")
                dataframe = dataframe.loc[
                    ((dataframe.ts_x - dataframe.ts_y).abs() < 24 * 60 * 60)
                    & (dataframe.aid_x != dataframe.aid_y)
                ]

                dataframe = dataframe.loc[
                    (dataframe.aid_x >= PART * TOTAL_SIZE)
                    & (dataframe.aid_x < (PART + 1) * TOTAL_SIZE)
                ]

                dataframe = dataframe[
                    ["session", "aid_x", "aid_y", "ts_x"]
                ].drop_duplicates(["session", "aid_x", "aid_y"])
                dataframe["wgt"] = 1 + 3 * (dataframe.ts_x - TS_BEGIN) / (
                    TS_END - TS_BEGIN
                )
                dataframe = dataframe[["aid_x", "aid_y", "wgt"]]
                dataframe.wgt = dataframe.wgt.astype("float32")
                dataframe = dataframe.groupby(["aid_x", "aid_y"]).wgt.sum()

                if k == a:
                    tmp2 = dataframe
                else:
                    tmp2 = tmp2.add(dataframe, fill_value=0)

            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, dataframe
            gc.collect()

        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])

        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < 20].drop("n", axis=1)
        tmp.to_parquet(f"{CLICKS_COVIS_FILE_PREFIX}_{PART}.pqt")


def carts_to_orders_covis(NUM_SECTIONS=4, TOTAL_SIZE=1.86e6 / 4):
    """
    Compute the co-visitation matrix for carts-to-orders interactions.

    This function processes the data in parts for memory management and generates the co-visitation matrix.
    It performs the following steps:
    1. Divide the data into 'NUM_SECTIONS' parts for efficient processing.
    2. Load the data in chunks and merge them in groups of 'READ_CT'.
    3. Use the tail of each session, keeping only up to 30 events per session.
    4. Create pairs of articles based on session information and timestamps.
    5. Assign weights to each pair based on the type of interaction and calculate the sum of weights for each pair.
    6. Combine the results from inner and outer chunks to form the co-visitation matrix.
    7. Convert the matrix to a dictionary and save the top 40 pairs for each article to disk.
    """
    print("Creating carts to orders covisitation matrix")

    for PART in range(NUM_SECTIONS):

        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))

            for k in range(a, b, READ_CT):

                dataframe = [read_file(files[k])]
                for i in range(1, READ_CT):
                    if k + i < b:
                        dataframe.append(read_file(files[k + i]))
                dataframe = pd.concat(dataframe, ignore_index=True, axis=0)
                dataframe = dataframe.sort_values(
                    ["session", "ts"], ascending=[True, False]
                )

                dataframe = dataframe.reset_index(drop=True)
                dataframe["n"] = dataframe.groupby("session").cumcount()
                dataframe = dataframe.loc[dataframe.n < 30].drop("n", axis=1)

                dataframe = dataframe.merge(dataframe, on="session")
                dataframe = dataframe.loc[
                    ((dataframe.ts_x - dataframe.ts_y).abs() < 24 * 60 * 60)
                    & (dataframe.aid_x != dataframe.aid_y)
                ]

                dataframe = dataframe.loc[
                    (dataframe.aid_x >= PART * TOTAL_SIZE)
                    & (dataframe.aid_x < (PART + 1) * TOTAL_SIZE)
                ]

                dataframe = dataframe[
                    ["session", "aid_x", "aid_y", "type_y"]
                ].drop_duplicates(["session", "aid_x", "aid_y"])
                dataframe["wgt"] = dataframe.type_y.map(TYPE_WEIGHT)
                dataframe = dataframe[["aid_x", "aid_y", "wgt"]]
                dataframe.wgt = dataframe.wgt.astype("float32")
                dataframe = dataframe.groupby(["aid_x", "aid_y"]).wgt.sum()

                if k == a:
                    tmp2 = dataframe
                else:
                    tmp2 = tmp2.add(dataframe, fill_value=0)

            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, dataframe
            gc.collect()

        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])

        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < 15].drop("n", axis=1)

        tmp.to_parquet(f"{CARTS_TO_ORDERS_COVIS_FILE_PREFIX}_{PART}.pqt")


def buy_to_buy_covis(NUM_SECTIONS=1, TOTAL_SIZE=1.86e6):
    """
    Compute the co-visitation matrix for buy-to-buy interactions.

    This function processes the data in parts for memory management and generates the co-visitation matrix.
    It performs the following steps:
    1. Divide the data into 'NUM_SECTIONS' parts for efficient processing.
    2. Load the data in chunks and merge them in groups of 'READ_CT'.
    3. Keep only the carts and orders interactions in the data.
    4. Use the tail of each session, keeping only up to 30 events per session.
    5. Create pairs of articles based on session information and timestamps.
    6. Filter pairs with a time difference of up to 14 days (2 weeks).
    7. Assign equal weights to each pair and calculate the sum of weights for each pair.
    8. Combine the results from inner and outer chunks to form the co-visitation matrix.
    9. Convert the matrix to a dictionary and save the top 40 pairs for each article to disk.

    Note:
    Some variable values (e.g., 'NUM_SECTIONS', 'TOTAL_SIZE', 'CHUNK', 'READ_CT', 'BUY_TO_BUY_COVIS_FILE_PREFIX', 'VER') are assumed
    to be defined elsewhere in the code.

    Example:
        buy_to_buy_covis()
    """
    print("Creating buy to buy covisitation matrix")

    for PART in range(NUM_SECTIONS):
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))

            for k in range(a, b, READ_CT):
                dataframe = [read_file(files[k])]
                for i in range(1, READ_CT):
                    if k + i < b:
                        dataframe.append(read_file(files[k + i]))
                dataframe = pd.concat(dataframe, ignore_index=True, axis=0)
                dataframe = dataframe.loc[dataframe["type"].isin([1, 2])]
                dataframe = dataframe.sort_values(
                    ["session", "ts"], ascending=[True, False]
                )

                dataframe = dataframe.reset_index(drop=True)
                dataframe["n"] = dataframe.groupby("session").cumcount()
                dataframe = dataframe.loc[dataframe.n < 30].drop("n", axis=1)

                dataframe = dataframe.merge(dataframe, on="session")
                dataframe = dataframe.loc[
                    ((dataframe.ts_x - dataframe.ts_y).abs() < 14 * 24 * 60 * 60)
                    & (dataframe.aid_x != dataframe.aid_y)
                ]

                dataframe = dataframe.loc[
                    (dataframe.aid_x >= PART * TOTAL_SIZE)
                    & (dataframe.aid_x < (PART + 1) * TOTAL_SIZE)
                ]

                dataframe = dataframe[
                    ["session", "aid_x", "aid_y", "type_y"]
                ].drop_duplicates(["session", "aid_x", "aid_y"])
                dataframe["wgt"] = 1
                dataframe = dataframe[["aid_x", "aid_y", "wgt"]]
                dataframe.wgt = dataframe.wgt.astype("float32")
                dataframe = dataframe.groupby(["aid_x", "aid_y"]).wgt.sum()

                if k == a:
                    tmp2 = dataframe
                else:
                    tmp2 = tmp2.add(dataframe, fill_value=0)

            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, dataframe
            gc.collect()

        tmp = tmp.reset_index()
        tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])

        tmp = tmp.reset_index(drop=True)
        tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
        tmp = tmp.loc[tmp.n < 15].drop("n", axis=1)

        tmp.to_parquet(f"{BUY_TO_BUY_COVIS_FILE_PREFIX}_{PART}.pqt")


def generate_covis_matrices():
    clicks_covis()
    carts_to_orders_covis()
    buy_to_buy_covis()
