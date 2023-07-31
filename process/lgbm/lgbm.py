import glob
import pandas as pd
import polars as pl
from lightgbm.sklearn import LGBMRanker

TRAIN_PATH = "../../test/resources/test.parquet"
TEST_PATH = "../../test/resources/test_parquet/*"
TEST_LABEL_PATH = "../../test/resources/test_labels.parquet"

TYPE_WEIGHTING_STR = {"clicks": 0, "carts": 1, "orders": 2}
TYPE_WEIGHTING = {0: 1, 1: 6, 2: 3}

PREDICTOR_VARS = [
    "aid",
    "type",
    "event_type_rev",
    "session_length",
    "ln_recency_score",
    "ln_recency_score_type_weighted",
]
RESPONSE_VAR = "gt"

lgbm_ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    n_estimators=20,
    importance_type="gain",
)

train = pl.read_parquet(TRAIN_PATH)
train_labels = pl.read_parquet(TEST_LABEL_PATH)


def event_type_rev(df):
    """
    Calculate the reverse cumulative count of events within each session in the DataFrame.

    This function calculates the reverse cumulative count of events within each session in the DataFrame. The reverse
    cumulative count represents the number of events in a session from the current row to the start of the session,
    i.e., counting events in reverse order within each session.

    Parameters:
    - df (DataFrame): The input DataFrame containing session data.

    Returns:
    - DataFrame: A new DataFrame with an additional column "event_type_rev" that contains the reverse cumulative count
                 of events within each session.
    """
    return df.select(
        [
            pl.col("*"),
            pl.col("session")
            .cumcount()
            .reverse()
            .over("session")
            .alias("event_type_rev"),
        ]
    )


def sessesion_len(df):
    """
    Calculate the length of each session in the DataFrame.

    This function calculates the length of each session in the DataFrame, i.e., the total number of events in each session.

    Parameters:
    - df (DataFrame): The input DataFrame containing session data.

    Returns:
    - DataFrame: A new DataFrame with an additional column "session_length" that contains the length of each session.
    """
    return df.select(
        [pl.col("*"), pl.col("session").count().over("session").alias("session_length")]
    )


def ln_recency_score(df):
    """
    Calculate the natural logarithm recency score for each row in the DataFrame.

    This function calculates the natural logarithm recency score for each row in the DataFrame based on the reverse
    cumulative count of events within each session and the session length. The recency score is used to prioritize
    recent events higher than older ones.

    Parameters:
    - df (DataFrame): The input DataFrame containing session data, including "session_length" and "event_type_rev" columns.

    Returns:
    - DataFrame: A new DataFrame with an additional column "ln_recency_score" that contains the natural logarithm
                 recency score for each row. If the score is NaN, it is filled with 1.
    """
    linear_interpolation = 0.1 + ((1 - 0.1) / (df["session_length"] - 1)) * (
        df["session_length"] - df["event_type_rev"] - 1
    )
    return df.with_columns(
        pl.Series(2 ** linear_interpolation - 1).alias("ln_recency_score")
    ).fill_nan(1)


def ln_recency_score_weighted(df):
    """
    Calculate the type-weighted natural logarithm recency score for each row in the DataFrame.

    This function calculates the type-weighted natural logarithm recency score for each row in the DataFrame.
    The type-weighted recency score is obtained by dividing the ln_recency_score by the corresponding type's weight
    based on the TYPE_WEIGHTING dictionary.

    Parameters:
    - df (DataFrame): The input DataFrame containing session data, including "ln_recency_score" and "type" columns.

    Returns:
    - DataFrame: A new DataFrame with an additional column "ln_recency_score_type_weighted" that contains the type-weighted
                 natural logarithm recency score for each row.
    """
    ln_recency_score_type_weighted = pl.Series(
        df["ln_recency_score"] / df["type"].apply(lambda x: TYPE_WEIGHTING[x])
    )
    return df.with_columns(
        ln_recency_score_type_weighted.alias("ln_recency_score_type_weighted")
    )


def apply(df, pipeline):
    """
    Apply a sequence of functions to the DataFrame in a pipeline fashion.

    This function applies a sequence of functions to the DataFrame in a pipeline fashion.
    Each function in the pipeline is applied to the DataFrame successively, and the result
    is passed as the input to the next function in the pipeline.

    Parameters:
    - df (DataFrame): The input DataFrame to be processed.
    - pipeline (list): A list of functions to be applied to the DataFrame in the specified order.

    Returns:
    - DataFrame: The DataFrame after applying all the functions in the pipeline to it.
    """
    for f in pipeline:
        df = f(df)
    return df


pipeline = [event_type_rev, sessesion_len, ln_recency_score, ln_recency_score_weighted]
train = apply(train, pipeline)

train_labels = train_labels.explode("ground_truth").with_columns(
    [
        pl.col("ground_truth").alias("aid"),
        pl.col("type").apply(lambda x: TYPE_WEIGHTING_STR[x]),
    ]
)[["session", "type", "aid"]]

train_labels = train_labels.with_columns(
    [
        pl.col("session").cast(pl.datatypes.Int32),
        pl.col("type").cast(pl.datatypes.UInt8),
        pl.col("aid").cast(pl.datatypes.Int32),
    ]
)

train_labels = train_labels.with_columns(pl.lit(1).alias("gt"))
train = train.join(
    train_labels, how="left", on=["session", "type", "aid"]
).with_columns(pl.col("gt").fill_null(0))


def get_session_lenghts(df):
    """
    Get the session lengths for each session in the DataFrame.

    This function calculates the session length (number of events) for each session in the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing session information.

    Returns:
    - numpy.ndarray: An array containing the session lengths for each session in the DataFrame.
    """
    return (
        df.groupby("session")
        .agg([pl.col("session").count().alias("session_length")])["session_length"]
        .to_numpy()
    )


session_lengths_train = get_session_lenghts(train)
lgbm_ranker = lgbm_ranker.fit(
    train[PREDICTOR_VARS].to_pandas(),
    train[RESPONSE_VAR].to_pandas(),
    group=session_lengths_train,
)


def load_test_files():
    """
    Load test files and concatenate them into a polars DataFrame.

    This function reads multiple parquet files from the TEST_PATH directory, processes the data, and concatenates
    them into a single polars DataFrame.

    Returns:
    - pl.DataFrame: The concatenated polars DataFrame containing the test data.
    """
    dfs = []
    for e, chunk_file in enumerate(glob.glob(TEST_PATH)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype("int32")
        chunk["type"] = (
            chunk["type"].map(TYPE_WEIGHTING_STR).astype("int8")
        )
        dfs.append(chunk)
    return pl.from_pandas(pd.concat(dfs).reset_index(drop=True))


session_types = []
labels = []

test = load_test_files()
test = apply(test, pipeline)
scores = lgbm_ranker.predict(test[PREDICTOR_VARS].to_pandas())
test = test.with_columns(pl.Series(name="score", values=scores))
test_predictions = test.sort(by=["session", "score"], descending=True)[
    ["session", "aid"]
]
test_predictions = (
    test_predictions.to_pandas()
    .groupby("session")
    .head(20)
    .groupby("session")
    .agg(list)
    .reset_index(drop=False)
)

for session, preds in zip(
    test_predictions["session"].to_numpy(), test_predictions["aid"].to_numpy()
):
    l = " ".join(str(p) for p in preds)
    for session_type in ["clicks", "carts", "orders"]:
        labels.append(l)
        session_types.append(f"{session}_{session_type}")

submission = pl.DataFrame({"session_type": session_types, "labels": labels})
submission.write_csv("submission.csv")
