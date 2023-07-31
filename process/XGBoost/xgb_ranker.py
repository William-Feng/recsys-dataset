import matplotlib.pyplot as plt
from merlin.core.utils import Distributed
from merlin.models.xgb import XGBoost
from merlin.schema.tags import Tags
from nvtabular import *
from nvtabular.ops import AddTags
import polars as pl
import xgboost as xgb

import helpers


TRAIN_PATH = "../../test/resources/test.parquet"
TRAIN_LABELS_PATH = "../../test/resources/test_labels.parquet"
TEST_PATH = "../../test/resources/test_full.parquet"
TYPE_LABELS = {"clicks": 0, "carts": 1, "orders": 2}


def preprocess_data(df, transformations):
    """Apply a list of transformations to a DataFrame (using the functions from helpers.py)."""

    for transform in transformations:
        df = transform(df)
    return df


def load_and_preprocess_data(pipeline):
    """Load and preprocess train and label data."""

    train_data = pl.read_parquet(TRAIN_PATH)
    train_labels = pl.read_parquet(TRAIN_LABELS_PATH)
    train_data = preprocess_data(train_data, pipeline)

    return train_data, train_labels


def convert_labels(df):
    """Converts the labels to appropriate format."""

    df = df.explode('ground_truth').with_columns([
        pl.col('ground_truth').alias('aid'),
        pl.col('type').apply(lambda x: TYPE_LABELS[x])
    ])[['session', 'type', 'aid']]

    df = df.with_columns([
        pl.col('session').cast(pl.datatypes.Int32),
        pl.col('type').cast(pl.datatypes.UInt8),
        pl.col('aid').cast(pl.datatypes.Int32)
    ])

    return df.with_columns(pl.lit(1).alias('gt'))


def build_model(train, labels):
    """Train the model on the given train data and labels using XGBoost with 3 different objectives."""

    train_labels = convert_labels(labels)

    train = train.join(train_labels, how='left', on=[
                       'session', 'type', 'aid']).with_columns(pl.col('gt').fill_null(0))

    data_set = Dataset(train.to_pandas())

    feature_cols = ['aid', 'type', 'session_duration', 'reversed_cumulative_count',
                    'session_recency_factor', 'weighted_session_recency_factor']
    target_col = ['gt'] >> AddTags([Tags.TARGET])
    id_col = ['session'] >> AddTags([Tags.USER_ID])

    wf = Workflow(feature_cols + target_col + id_col)
    train_processed = wf.fit_transform(data_set)

    objectives = ['rank:ndcg', 'rank:map', 'rank:pairwise']
    rankers = []

    for objective in objectives:
        ranker = XGBoost(train_processed.schema, objective=objective)
        with Distributed():
            ranker.fit(train_processed)
        rankers.append(ranker)

    return rankers, wf


def plot_feature_importance(ranker):
    """Plot feature importance of the trained model."""

    ranker.booster.save_model('xgb_model.json')
    bst = xgb.Booster()
    bst.load_model('xgb_model.json')

    fig, ax = plt.subplots(figsize=(16, 8))
    xgb.plot_importance(bst, ax=ax)
    plt.tight_layout()
    plt.savefig('feature_importance.png')


def make_predictions(rankers, wf, pipeline, weights):
    """Make predictions on test data using the weighted model."""

    test_data = pl.read_parquet(TEST_PATH)
    test_data = preprocess_data(test_data, pipeline)
    data_set = Dataset(test_data.to_pandas())

    wf = wf.remove_inputs(['gt'])
    test_processed = wf.transform(data_set)

    test_preds = 0
    for ranker, weight in zip(rankers, weights):
        test_preds += weight * \
            ranker.booster.predict(xgb.DMatrix(test_processed.compute()))

    test_data = test_data.with_columns(
        pl.Series(name='score', values=test_preds))

    test_predictions = test_data.sort(['session', 'score'], descending=True).groupby('session').agg([
        pl.col('aid').apply(list).alias('aid_list')
    ])

    test_predictions = test_predictions.with_columns(
        pl.col('aid_list').apply(lambda x: x[:20] if len(x) > 20 else x)
    )

    return test_predictions


def create_submission_file(test_predictions, weights):
    """Create a submission file for all weights from the test predictions and ensure they're in the correct format."""

    session_types = []
    labels = []

    for session, preds in zip(test_predictions['session'].to_numpy(), test_predictions['aid_list'].to_numpy()):
        aid = ' '.join(str(pred) for pred in preds)
        for session_type in ['clicks', 'carts', 'orders']:
            labels.append(aid)
            session_types.append(f'{session}_{session_type}')

    submission = pl.DataFrame(
        {'session_type': session_types, 'labels': labels})

    submission = submission.with_columns([
        submission['session_type'].apply(lambda x: int(
            x.split('_')[0]), return_dtype=pl.Int64).alias('session_num'),
        submission['session_type'].apply(
            lambda x: x.split('_')[1]).alias('session_action')
    ])

    submission = submission.with_columns([
        submission['session_action'].apply(
            lambda x: TYPE_LABELS[x]).alias('action_order')
    ])
    submission = submission.sort(['session_num', 'action_order'])
    submission = submission.drop(
        ['session_num', 'action_order', 'session_action'])

    weight_1 = "{:.1f}".format(weights[0])
    weight_2 = "{:.1f}".format(weights[1])
    weight_3 = "{:.1f}".format(weights[2])

    submission.write_csv(f'predictions/{weight_1}_{weight_2}_{weight_3}.csv')


def get_weights():
    """Get all possible weights for the 3 rankers (ensure that the total sum is 1)."""

    step = 0.1
    weights_list = []
    for i in range(0, 11):
        for j in range(0, 11 - i):
            k = 10 - i - j
            num1 = i * step
            num2 = j * step
            num3 = k * step
            if num1 + num2 + num3 == 1.0:
                weights_list.append([num1, num2, num3])
    return weights_list


def main():
    """Main function to run the model and determine the best weights."""

    pipeline = helpers.get_pipeline()
    train_data, train_labels = load_and_preprocess_data(pipeline)
    rankers, workflow = build_model(train_data, train_labels)

    # Uncomment the following line to plot feature importance for a single desired ranker
    # plot_feature_importance(ranker)

    for weights in get_weights():
        predictions = make_predictions(rankers, workflow, pipeline, weights)
        create_submission_file(predictions, weights)


if __name__ == "__main__":
    main()
