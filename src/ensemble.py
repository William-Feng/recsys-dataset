# COMP9417 Group Project
# Ensembling models
# Authored 31/07/23

import polars as pl

def read_sub(fp, weight): 
    '''
    Loads the submission files to iterate through

    Typecasting is used to retain memory through the operations
    also typecasts article id values to int32 values to optimise memory

    Params:
        fp (str): a given file path for the submission file
        weight: given weightage value for the model in question
    
    Returns:
        dataframe (polars): containing aid values
    '''
    return (
        pl.read_csv(fp)
            .with_columns(pl.col('labels').str.split(by=' ')).with_columns(pl.lit(weight).alias('v'))
            .explode('labels').rename({'labels': 'aid'})
            .with_columns(pl.col('aid').cast(pl.UInt32)).with_columns(pl.col('v').cast(pl.UInt8))
    )

def perform_ensemble(model_predictions):
    '''
    Carries out voter ensemble

    Creates df based on predictions from the given models

    Params:
        model_predictions (list): contains the predictions from given models

    Returns:
        final_predictions (df): ensemble predictions after voting ensemble
    '''
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
    final_predictions = final_predictions.with_columns(pl.col('labels').cast(pl.List(pl.Utf8)).list.join(' '))
    return final_predictions

# Store the prediction filepaths
prediction_fps = ['./../ensemble-submissions/w2v_predictions.csv', './../ensemble-submissions/xgb_predictions.csv',
         './../ensemble-submissions/covis_predictions.csv']
# Applying greater weights to better models 
model_predictions = [read_sub(fp, weight_val) for fp, weight_val in zip(prediction_fps, [0.2, 0.3, 0.5])]
# Perform the voter ensemble method
final_predictions = perform_ensemble(model_predictions)
# Place final predictions in submission file
final_predictions.write_csv('submission.csv')
