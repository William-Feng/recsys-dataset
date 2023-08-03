"""
This module is for exploratory data analysis (EDA) purposes
"""


import pandas as pd
from pathlib import Path


PATH = Path("../../test/resources")


def build_event_df_from_chunks(chunk):
    """
    This function constructs a DataFrame of event data from a chunk of data.

    Params:
        chunk (DataFrame): Chunk of data to process.

    Returns:
        DataFrame: Event data.
    """

    event_dict = {"session": [], "aid": [], "ts": [], "type": []}

    for session, events in zip(chunk["session"].tolist(), chunk["events"].tolist()):
        for event in events:
            event_dict["session"].append(session)
            event_dict["aid"].append(event["aid"])
            event_dict["ts"].append(event["ts"])
            event_dict["type"].append(event["type"])

    return pd.DataFrame(event_dict)


def load_data(file_path, sample_size=100_000):
    """
    This function loads JSON data in chunks from a specified file path, 
    and consolidates the chunks into a DataFrame.

    Params:
        file_path (Path): Path to the JSON file.
        sample_size (int, optional): Size of the chunk for reading the file. Defaults to 100_000.

    Returns:
        DataFrame: Consolidated data from all chunks.
    """

    df = pd.DataFrame()
    chunks = pd.read_json(file_path, lines=True, chunksize=sample_size)

    for chunk in chunks:
        df = pd.concat([df, build_event_df_from_chunks(chunk)])

    return df


def get_best_sold_list(df, top_n=20):
    """
    This function creates a list of best sold items.

    Params:
        df (DataFrame): Data containing 'type', 'aid', and 'count' columns.
        top_n (int, optional): Number of top items to include. Defaults to 20.

    Returns:
        str: Concatenated list of top sold item aids.
    """

    order_num_df = (
        df.groupby(["type", "aid"])["session"]
        .agg("count")
        .reset_index()
        .rename(columns={"session": "count"})
    )
    order_num_df = order_num_df[order_num_df["type"] == "orders"]
    order_num_df = order_num_df.sort_values(["count"], ascending=False).reset_index(
        drop=True
    )

    return " " + order_num_df[:top_n].aid.astype("str").sum()


def get_recommended(df, action_type):
    """
    This function creates a DataFrame of recommended actions.

    Params:
        df (DataFrame): DataFrame to extract actions from.
        action_type (str): Type of action to extract ('clicks', 'carts', 'orders').

    Returns:
        DataFrame: Recommended actions of a specified type.
    """

    df = df[df["type"] == action_type].copy()
    df["type"] = action_type

    return df


def create_submission(recommend_df, best_sold_list, data_path):
    """
    This function creates a submission file based on the provided recommendations and best sold list.

    Params:
        recommend_df (DataFrame): DataFrame of recommendations.
        best_sold_list (str): String of best sold item aids.
        data_path (Path): Path to the sample submission file.
    """

    sample_sub = pd.read_csv(data_path / "sample_submission.csv")
    sample_sub = pd.merge(
        sample_sub, recommend_df[["session_type", "aid"]], on="session_type", how="left"
    )
    sample_sub["next"] = sample_sub["aid"].fillna("") + best_sold_list
    sample_sub["next"] = sample_sub["next"].str.strip()
    sample_sub.drop(["labels", "aid"], axis=1, inplace=True)
    sample_sub.columns = ("session_type", "labels")
    sample_sub.to_csv("predictions.csv", index=False)


def main():
    """
    Main function to orchestrate the data loading, displaying, processing and submission creation.
    """

    pd.set_option("display.max_columns", None)

    train_df = load_data(PATH / "otto-recsys-train.jsonl")
    train_df["minutes"] = train_df.groupby("session")["ts"].diff(-1) * (-1 / 1000 / 60)
    best_sold_list = get_best_sold_list(train_df)

    test_df = load_data(PATH / "otto-recsys-test.jsonl")
    test_df["minutes"] = test_df.groupby("session")["ts"].diff(-1) * (-1 / 1000 / 60)

    test_action_df = test_df.copy()
    test_action_df["aid"] = " " + test_df["aid"].astype("str")
    test_action_df = (
        test_action_df.groupby(["session", "type"])["aid"].sum().reset_index()
    )

    next_clicks_df = get_recommended(test_action_df, "clicks")
    next_orders_df = get_recommended(test_action_df, "carts")
    next_orders_df = pd.merge(
        next_orders_df, next_clicks_df[["session", "aid"]], on="session", how="left"
    )
    next_orders_df["aid"] = next_orders_df["aid_x"] + next_orders_df["aid_y"]
    next_orders_df.drop(["aid_x", "aid_y"], axis=1, inplace=True)

    next_carts_df = get_recommended(test_action_df, "clicks")

    recommend_df = pd.concat([next_orders_df, next_carts_df, next_clicks_df], axis=0)
    recommend_df["session_type"] = (
        recommend_df["session"].astype("str") + "_" + recommend_df["type"]
    )

    create_submission(recommend_df, best_sold_list, PATH)


if __name__ == "__main__":
    main()
