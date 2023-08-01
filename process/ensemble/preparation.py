"""
Since the predictions.csv file is not necessarily sorted by session_type, 
this module is used to firstly sort it by the session_id, then by the session_type.

This pre-processing allows the weighted_ensemble.py module to easily read through the predictions.csv
files simultaneously generated by the various models, and then perform the weighted ensemble.
"""


import pandas as pd


def main():
    """
    Performs a series of transformations on input data, and then outputs the sorted data.
    """

    # Load your CSV file
    df = pd.read_csv('../../ensemble-submissions/predictions.csv')

    # Split the 'session_type' column into two separate columns
    df[['session_id', 'session_type']
       ] = df['session_type'].str.split('_', expand=True)

    # Convert 'session_id' to int for proper sorting
    df['session_id'] = df['session_id'].astype(int)

    # Create a dictionary to map session_types to integers
    session_type_dict = {"clicks": 0, "carts": 1, "orders": 2}

    # Map the 'session_type' to its corresponding integer
    df['session_type'] = df['session_type'].map(session_type_dict)

    # Sort the DataFrame
    df = df.sort_values(by=['session_id', 'session_type'])

    # Concatenate 'session_id' and 'session_type' back into a single column
    df['session_type'] = df['session_id'].astype(
        str) + '_' + df['session_type'].map({v: k for k, v in session_type_dict.items()})

    # Drop the 'session_id' column
    df = df.drop(columns='session_id')

    # Write the DataFrame back to CSV
    df.to_csv('predictions_sorted.csv', index=False)


if __name__ == "__main__":
    main()
