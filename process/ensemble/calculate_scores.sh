#!/bin/bash

# This script will calculate the score of all the generated prediction files and store them in output.log

# The weights [0.80, 0.35, 0.30, 0.55] for Word2Vec, Re-ranker, EDA, XGBoost respectively is best.
# Note that the weights normalised is [0.40, 0.175, 0.15, 0.275].
# This gave the highest recall@20 score of 0.92643 and a Kaggle private score of 0.89638.


# Run this file from the src directory
run_dir='../../src/'
cd $run_dir

# Target directory
dir='../process/ensemble/predictions/'

# Output file
outfile='../process/ensemble/output.log'

# Remove the output file if it already exists
if [ -f $outfile ] ; then
    rm $outfile
fi

# Iterate over all CSV files in the target directory
for file in $dir*.csv
do
  # Run the testing command, filter out the desired score line and remove 'INFO:root:' part
  line=$(pipenv run python -m evaluate --test-labels out/test_labels.jsonl --predictions "$file" 2>&1 | grep 'Scores' | sed 's/INFO:root://')

  # Append the filename and the line to the output file
  echo "$(basename "$file")" >> $outfile
  echo "$line" >> $outfile
  echo "" >> $outfile
done
