#!/bin/bash

# This script will calculate the score of all the generated prediction files and store them in output.log

# The weights [0.5, 0.2, 0.3] for [ndcg, map, pairwise] XGBoost Objective Rank respectively is best
# This gave the highest recall@20 score of 0.595991438497308



# Run this file from the src directory
run_dir='../../src/'
cd $run_dir

# Target directory
dir='../process/XGBoost/predictions/'

# Output file
outfile='../process/XGBoost/output.log'

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
