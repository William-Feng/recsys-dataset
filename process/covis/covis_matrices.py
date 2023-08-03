"""
This module is used for stage one in generating the Covisitation Matrices.
"""


import pandas as pd
import gc

from utils import (
    CHUNK,
    MAX_DISK_SIZE,
    READ_CT,
    TIME_MS_24_HRS,
    TS_BEGIN,
    TS_END,
    TYPE_WEIGHT,
    files,
    read_file,
)


def clicks_covis(NUM_SECTIONS=4, TOTAL_SIZE=MAX_DISK_SIZE / 4):
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
    7. Convert the matrix to total_chunks dictionary and save the top 40 pairs for each article to disk.
    """
    print("Creating clicks covisitation matrix")

    def get_matrix(dataframe):
        dataframe = pd.concat(dataframe, ignore_index=True, axis=0)
        dataframe = dataframe.sort_values(
            ["session", "ts"], ascending=[True, False]
        )

        dataframe = dataframe.reset_index(drop=True)
        dataframe["n"] = dataframe.groupby("session").cumcount()
        dataframe = dataframe.loc[dataframe.n < 30].drop("n", axis=1)

        dataframe = dataframe.merge(dataframe, on="session")
        dataframe = dataframe.loc[
            ((dataframe.ts_x - dataframe.ts_y).abs() < TIME_MS_24_HRS)
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
        return dataframe

    for PART in range(NUM_SECTIONS):
        for i in range(6):
            total_chunks = i * CHUNK
            b = min((i + 1) * CHUNK, len(files))

            for disk_size in range(total_chunks, b, READ_CT):
                dataframe = [read_file(files[disk_size])]
                for i in range(1, READ_CT):
                    if disk_size + i < b:
                        dataframe.append(read_file(files[disk_size + i]))
                dataframe = get_matrix(dataframe)

                curr_dataframe = dataframe if disk_size == total_chunks else  curr_dataframe.add(dataframe, fill_value=0)
            new_curr_dataframe = curr_dataframe if total_chunks == 0 else new_curr_dataframe.add(curr_dataframe, fill_value=0)
            del curr_dataframe, dataframe
            gc.collect()

        new_curr_dataframe = new_curr_dataframe.reset_index()
        new_curr_dataframe = new_curr_dataframe.sort_values(["aid_x", "wgt"], ascending=[True, False])

        new_curr_dataframe = new_curr_dataframe.reset_index(drop=True)
        new_curr_dataframe["n"] = new_curr_dataframe.groupby("aid_x").aid_y.cumcount()
        new_curr_dataframe = new_curr_dataframe.loc[new_curr_dataframe.n < 20].drop("n", axis=1)
        new_curr_dataframe.to_parquet(f"{CLICKS_COVIS_FILE_PREFIX}_{PART}.pqt")


def carts_to_orders_covis(NUM_SECTIONS=4, TOTAL_SIZE=MAX_DISK_SIZE / 4):
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
    7. Convert the matrix to total_chunks dictionary and save the top 40 pairs for each article to disk.
    """
    print("Creating carts to orders covisitation matrix")

    def get_matrix(dataframe):
        dataframe = pd.concat(dataframe, ignore_index=True, axis=0)
        dataframe = dataframe.sort_values(
            ["session", "ts"], ascending=[True, False]
        )

        dataframe = dataframe.reset_index(drop=True)
        dataframe["n"] = dataframe.groupby("session").cumcount()
        dataframe = dataframe.loc[dataframe.n < 30].drop("n", axis=1)

        dataframe = dataframe.merge(dataframe, on="session")
        dataframe = dataframe.loc[
            ((dataframe.ts_x - dataframe.ts_y).abs() < TIME_MS_24_HRS)
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
        return dataframe

    for PART in range(NUM_SECTIONS):
        for i in range(6):
            total_chunks = i * CHUNK
            b = min((i + 1) * CHUNK, len(files))

            for disk_size in range(total_chunks, b, READ_CT):

                dataframe = [read_file(files[disk_size])]
                for i in range(1, READ_CT):
                    if disk_size + i < b:
                        dataframe.append(read_file(files[disk_size + i]))
                dataframe = get_matrix(dataframe)

                curr_dataframe = dataframe if disk_size == total_chunks else  curr_dataframe.add(dataframe, fill_value=0)
            new_curr_dataframe = curr_dataframe if total_chunks == 0 else new_curr_dataframe.add(curr_dataframe, fill_value=0)
            del curr_dataframe, dataframe
            gc.collect()

        new_curr_dataframe = new_curr_dataframe.reset_index()
        new_curr_dataframe = new_curr_dataframe.sort_values(["aid_x", "wgt"], ascending=[True, False])

        new_curr_dataframe = new_curr_dataframe.reset_index(drop=True)
        new_curr_dataframe["n"] = new_curr_dataframe.groupby("aid_x").aid_y.cumcount()
        new_curr_dataframe = new_curr_dataframe.loc[new_curr_dataframe.n < 15].drop("n", axis=1)

        new_curr_dataframe.to_parquet(f"{CARTS_TO_ORDERS_COVIS_FILE_PREFIX}_{PART}.pqt")


def buy_to_buy_covis(NUM_SECTIONS=1, TOTAL_SIZE=MAX_DISK_SIZE):
    """
    Compute the co-visitation matrix for buy-to-buy interactions.

    This function processes the data in parts for memory management and generates the co-visitation matrix.
    It performs the following steps:
    1. Divide the data into 'NUM_SECTIONS' parts for efficient processing.
    2. Load the data in chunks and merge them in groups of 'READ_CT'.
    3. Keep only the carts and orders interactions in the data.
    4. Use the tail of each session, keeping only up to 30 events per session.
    5. Create pairs of articles based on session information and timestamps.
    6. Filter pairs with total_chunks time difference of up to 14 days (2 weeks).
    7. Assign equal weights to each pair and calculate the sum of weights for each pair.
    8. Combine the results from inner and outer chunks to form the co-visitation matrix.
    9. Convert the matrix to total_chunks dictionary and save the top 40 pairs for each article to disk.

    Note:
    Some variable values (e.g., 'NUM_SECTIONS', 'TOTAL_SIZE', 'CHUNK', 'READ_CT', 'BUY_TO_BUY_COVIS_FILE_PREFIX', 'VER') are assumed
    to be defined elsewhere in the code.

    Example:
        buy_to_buy_covis()
    """
    print("Creating buy to buy covisitation matrix")

    def get_matrix(dataframe):
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
            ((dataframe.ts_x - dataframe.ts_y).abs() < 14 * TIME_MS_24_HRS)
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
        
        return dataframe

    for PART in range(NUM_SECTIONS):
        for i in range(6):
            total_chunks = i * CHUNK
            b = min((i + 1) * CHUNK, len(files))

            for disk_size in range(total_chunks, b, READ_CT):
                dataframe = [read_file(files[disk_size])]
                for i in range(1, READ_CT):
                    if disk_size + i < b:
                        dataframe.append(read_file(files[disk_size + i]))
                dataframe = get_matrix(dataframe)

                curr_dataframe = dataframe if disk_size == total_chunks else  curr_dataframe.add(dataframe, fill_value=0)
            new_curr_dataframe = curr_dataframe if total_chunks == 0 else new_curr_dataframe.add(curr_dataframe, fill_value=0)
            del curr_dataframe, dataframe
            gc.collect()

        new_curr_dataframe = new_curr_dataframe.reset_index()
        new_curr_dataframe = new_curr_dataframe.sort_values(["aid_x", "wgt"], ascending=[True, False])

        new_curr_dataframe = new_curr_dataframe.reset_index(drop=True)
        new_curr_dataframe["n"] = new_curr_dataframe.groupby("aid_x").aid_y.cumcount()
        new_curr_dataframe = new_curr_dataframe.loc[new_curr_dataframe.n < 15].drop("n", axis=1)

        new_curr_dataframe.to_parquet(f"{BUY_TO_BUY_COVIS_FILE_PREFIX}_{PART}.pqt")


def generate_covis_matrices():
    clicks_covis()
    carts_to_orders_covis()
    buy_to_buy_covis()
