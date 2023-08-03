"""
This module is used for stage two in re-ranking the Covisitation Matrices (along with getting the submission)
"""


from collections import Counter
import glob
import itertools
import numpy as np

import pandas as pd
from utils import (
    PREDICTIONS_OUTPUT_PATH,
    TEST_PATH,
    TOP_N,
    TYPE_LABELS,
    TYPE_WEIGHT,
    get_covis_matrices_dict,
)

NUM_SECTION = 4
TOTAL_SIZE = 2000000 / NUM_SECTION


def load_test():
    """
    Load and concatenate test data from parquet files.

    Returns:
    pandas.DataFrame: Concatenated test data with 'ts', 'type', and other features.
    """
    dataframes = []
    for e, chunk_file in enumerate(glob.glob(TEST_PATH)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype("int32")
        chunk["type"] = chunk["type"].map(TYPE_LABELS).astype("int8")
        dataframes.append(chunk)
    return pd.concat(dataframes).reset_index(drop=True)


def suggest_clicks(dataframe, clicks_covis, top_clicks):
    """
    Generate a list of suggested article IDs for user clicks based on the provided data.

    Params:
    dataframe (pandas.DataFrame):
        Input DataFrame containing user history with columns 'aid' (article ID) and 'type' (event type).
    clicks_covis (dict):
        Co-visitation matrix containing article ID co-occurrence information for clicks.
    top_clicks (list):
        List of top TOP_N test clicks.

    Returns:
    list: A list of suggested article IDs for user clicks.

    Note:
    The function first re-ranks the candidates based on the user history by considering weights derived
    from the type of items visited and their repeat occurrences. If there are at least TOP_N unique items,
    the re-ranked list is returned. Otherwise, the co-visitation matrix is used to recommend additional
    article IDs to complete the list. Finally, the top TOP_N test clicks are used to further augment the list.
    """
    aids = dataframe.aid.tolist()
    types = dataframe.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    if len(unique_aids) >= TOP_N:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * TYPE_WEIGHT[t]
        sorted_aids = [k for k, v in aids_temp.most_common(TOP_N)]
        return sorted_aids
    aids2 = list(
        itertools.chain(
            *[clicks_covis[aid] for aid in unique_aids if aid in clicks_covis]
        )
    )
    top_aids2 = [
        aid2
        for aid2, cnt in Counter(aids2).most_common(TOP_N)
        if aid2 not in unique_aids
    ]
    result = unique_aids + top_aids2[: TOP_N - len(unique_aids)]
    return result + list(top_clicks)[: TOP_N - len(result)]


def suggest_buys(dataframe, buy_to_buy_covis, carts_to_orders_covis, top_orders):
    """
    Generate suggested article IDs for user buys.

    Params:
    dataframe (pandas.DataFrame): User history with 'aid' and 'type' columns.
    buy_to_buy_covis (dict): Co-visitation matrix for buy-to-buy interactions.
    carts_to_orders_covis (dict): Co-visitation matrix for cart-to-order interactions.
    top_orders (list): Top TOP_N test orders.

    Returns:
    list: Suggested article IDs for user buys.
    """
    aids = dataframe.aid.tolist()
    types = dataframe.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    dataframe = dataframe.loc[(dataframe["type"] == 1) | (dataframe["type"] == 2)]
    unique_buys = list(dict.fromkeys(dataframe.aid.tolist()[::-1]))
    if len(unique_aids) >= TOP_N:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * TYPE_WEIGHT[t]
        aids3 = list(
            itertools.chain(
                *[
                    buy_to_buy_covis[aid]
                    for aid in unique_buys
                    if aid in buy_to_buy_covis
                ]
            )
        )
        for aid in aids3:
            aids_temp[aid] += 0.1
        sorted_aids = [k for k, v in aids_temp.most_common(TOP_N)]
        return sorted_aids
    aids2 = list(
        itertools.chain(
            *[
                carts_to_orders_covis[aid]
                for aid in unique_aids
                if aid in carts_to_orders_covis
            ]
        )
    )
    aids3 = list(
        itertools.chain(
            *[buy_to_buy_covis[aid] for aid in unique_buys if aid in buy_to_buy_covis]
        )
    )
    top_aids2 = [
        aid2
        for aid2, cnt in Counter(aids2 + aids3).most_common(TOP_N)
        if aid2 not in unique_aids
    ]
    result = unique_aids + top_aids2[: TOP_N - len(unique_aids)]
    return result + list(top_orders)[: TOP_N - len(result)]


def rerank():
    """
    Rerank test data using co-visitation matrices and generate prediction output.

    This function performs the following steps:
    1. Loads test data using the 'load_test' function.
    2. Loads three co-visitation matrices: 'clicks_covis', 'carts_to_orders_covis', and 'buy_to_buy_covis'.
    3. Retrieves the top clicks and top orders from the test data.
    4. Reranks the test data for clicks and buys using the co-visitation matrices and the 'suggest_clicks' and 'suggest_buys' functions.
    5. Concatenates the results for clicks, orders, and carts into a single DataFrame 'pred_dataframe'.
    6. Converts the 'labels' column in 'pred_dataframe' to a space-separated string.
    7. Writes the final prediction output to 'PREDICTIONS_OUTPUT_PATH'.

    Note:
    The 'suggest_clicks' and 'suggest_buys' functions are assumed to be defined separately.
    """
    test_dataframe = load_test()

    matrices = get_covis_matrices_dict()
    clicks_covis = matrices["click_covis"]
    carts_to_orders_covis = matrices["carts_to_orders_covis"]
    buy_to_buy_covis = matrices["buy_to_buy_covis"]

    top_clicks = (
        test_dataframe.loc[test_dataframe["type"] == "clicks", "aid"]
        .value_counts()
        .index.values[:TOP_N]
    )
    top_orders = (
        test_dataframe.loc[test_dataframe["type"] == "orders", "aid"]
        .value_counts()
        .index.values[:TOP_N]
    )

    pred_dataframe_clicks = (
        test_dataframe.sort_values(["session", "ts"])
        .groupby(["session"])
        .apply(lambda x: suggest_clicks(x, clicks_covis, top_clicks))
    )

    pred_dataframe_buys = (
        test_dataframe.sort_values(["session", "ts"])
        .groupby(["session"])
        .apply(
            lambda x: suggest_buys(
                x, buy_to_buy_covis, carts_to_orders_covis, top_orders
            )
        )
    )

    clicks_pred_dataframe = pd.DataFrame(
        pred_dataframe_clicks.add_suffix("_clicks"), columns=["labels"]
    ).reset_index()
    orders_pred_dataframe = pd.DataFrame(
        pred_dataframe_buys.add_suffix("_orders"), columns=["labels"]
    ).reset_index()
    carts_pred_dataframe = pd.DataFrame(
        pred_dataframe_buys.add_suffix("_carts"), columns=["labels"]
    ).reset_index()

    pred_dataframe = pd.concat(
        [clicks_pred_dataframe, orders_pred_dataframe, carts_pred_dataframe]
    )
    pred_dataframe.columns = ["session_type", "labels"]
    pred_dataframe["labels"] = pred_dataframe.labels.apply(
        lambda x: " ".join(map(str, x))
    )
    pred_dataframe.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
