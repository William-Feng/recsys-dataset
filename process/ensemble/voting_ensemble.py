"""
This module performs voter ensemble on the predictions from three models used in the project.

Note that since this performed poorly, it was not used at all for the submission.
The best submission came from weighted_ensemble.py.
"""


import polars as pl


PATH = '../../ensemble-submissions'


def read_submission(fp, weight):
    """
    Loads the submission files to iterate through.

    Typecasting is used to retain and optimise memory through the operations.
    For example, the labels column is split by spaces and article_id is typecasted to int32 values.

    Params:
        fp (str): File path of the submission file.
        weight: The weight to assign to the submission during ensemble processing.

    Returns:
        dataframe (polars): A DataFrame that includes the submission's 'aid' (article ID) values and associated weights.
    """

    return (
        pl.read_csv(fp)
        .with_columns(pl.col('labels').str.split(by=' ')).with_columns(pl.lit(weight).alias('v'))
        .explode('labels').rename({'labels': 'aid'})
        .with_columns(pl.col('aid').cast(pl.UInt32)).with_columns(pl.col('v').cast(pl.UInt8))
    )


def perform_ensemble(model_predictions):
    """
    Carries out voter ensemble.
    Creates df based on predictions from the given models.

    Params:
        model_predictions (list): A list of DataFrames containing predictions from different models.

    Returns:
        final_predictions (df): A DataFrame containing the final ensemble predictions.
    """

    model_predictions = model_predictions[0].join(model_predictions[1],
                                                  how='outer',
                                                  on=['session_type',
                                                      'aid']).join(model_predictions[2],
                                                                   how='outer', on=['session_type', 'aid'],
                                                                   suffix='r2')
    # Fill gaps (values that are currently null)
    model_predictions = (model_predictions
                         .fill_null(0).with_columns((pl.col('v') + pl.col('v_right') + pl.col('vr2')).alias('sum_votes'))
                         .drop(['v', 'v_right', 'vr2']).sort(by='sum_votes').reverse()
                         )
    # Format
    final_predictions = model_predictions.groupby('session_type').agg([
        pl.col('aid').head(20).alias('labels')
    ])
    # Casting the numbers to improve speed
    final_predictions = final_predictions.with_columns(
        pl.col('labels').cast(pl.List(pl.Utf8)).list.join(' '))
    return final_predictions


def main():
    """
    Main function that orchestrates the reading of model predictions and their ensemble
    into final predictions. 

    Defines the file paths and associated weights for each model prediction. Then, reads
    these prediction files and performs the ensemble operation on them. The final ensemble
    predictions are written to a csv file.
    """

    prediction_fps = [f'{PATH}/w2v_predictions.csv', f'{PATH}/xgb_predictions.csv',
                      f'{PATH}/covis_predictions.csv']
    model_predictions = [read_submission(fp, weight_val)
                         for fp, weight_val in zip(prediction_fps, [0.4, 0.3, 0.5])]
    final_predictions = perform_ensemble(model_predictions)
    final_predictions.write_csv('submission.csv')


if __name__ == '__main__':
    main()
