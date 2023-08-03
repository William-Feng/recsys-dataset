"""
This module is used to combine the predictions from the 5 models using weighted average ensemble.
These 5 prediction files came from:
- Word2Vec Model
- Re-ranker Model
- Exploratory Data Analysis (EDA)
- XGBoost Model
- LightGBM Model (Not used in final submission after LOCCV)

This file was used to create a series of submissions with different weights to determine the best combination.

The final normalised weights used in the submission were: [0.40, 0.175, 0.15, 0.275]
"""


from collections import Counter
from itertools import product
import numpy as np
import pandas as pd


PATH = '../../ensemble-submissions'
file_names = [f'{PATH}/eda_predictions.csv', f'{PATH}/covis_predictions.csv',
              f'{PATH}/w2v_predictions.csv', f'{PATH}/xgb_predictions.csv']


def get_weights():
    """
    Gets all possible weight combinations for the 4 models.
    We also want to filter out combinations that don't sum to a total of 2.0.
    To avoid overfitting or too many models being generated, an individual weighting must be in [0.2, 0.8]
    """

    weights = np.arange(0.20, 0.85, 0.05)
    all_combinations = product(weights, repeat=4)

    return [comb for comb in all_combinations if sum(comb) == 2.0]


def read_file(file_name):
    """
    Reads a CSV file and convert label strings to lists of integers.

    Params:
        file_name (str): The name of the file to read.

    Returns:
        pd.DataFrame: The dataframe containing the data from the CSV file with labels as list of integers.
    """

    df = pd.read_csv(file_name)
    df['labels'] = df['labels'].apply(lambda x: list(map(int, str(x).split())))
    return df


def combine_labels(labels_list, weights):
    """
    Combines multiple label lists into a single list using the given weights.
    The weights are used to prioritise the labels from different models.
    The priority of labels within a single model decays with a factor of 0.03 for each subsequent label.
    Finally, the labels are sorted by their weighted counts in descending order and the top 20 are taken.

    Params:
        labels_list (List[List[int]]): The list of label lists to be combined.
        weights (List[float]): The weights for each label list.

    Returns:
        List[int]: The combined label list.
    """

    counter_sum = Counter()

    for labels, weight in zip(labels_list, weights):
        weighted_counter = Counter()
        for i, label in enumerate(labels):
            weighted_counter[label] += weight * (0.97 ** i)
        counter_sum += weighted_counter

    top_labels = counter_sum.most_common(20)

    return [label for label, _ in top_labels]


def main():
    """
    Main function of the script that coordinates the reading, combining and writing processes.

    For each valid combination of weights, this function:
        - Reads the prediction files
        - Combines the predictions from the different files using the given weights
        - Converts the combined lists of labels back to space-separated strings
        - Writes the combined predictions to a new CSV file named after the weights used
    """

    for weights in get_weights():
        dfs = [read_file(file_name) for file_name in file_names]
        combined_df = dfs[0].copy()
        combined_df['labels'] = list(map(combine_labels, zip(
            *[df['labels'] for df in dfs]), [weights]*len(dfs[0])))

        combined_df['labels'] = combined_df['labels'].apply(
            lambda x: ' '.join(map(str, x)))

        formatted_weights = [f"{weight:.2f}" for weight in weights]
        prediction_file = "_".join(formatted_weights) + ".csv"
        combined_df.to_csv(
            "predictions/" + prediction_file, index=False)

        print(f'Processed {prediction_file}')


if __name__ == '__main__':
    main()
