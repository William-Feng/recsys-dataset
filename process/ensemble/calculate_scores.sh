#!/bin/bash

# This script will calculate the score of all the generated prediction files and store them in output.log

# The weights TODO for EDA, Covisitation Matrices, Word2Vec, XGBoost respectively is best
# This gave the highest recall@20 score of TODO


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
