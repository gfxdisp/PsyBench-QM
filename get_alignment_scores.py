import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
from pyst.ContrastMatching import ContrastConstancySpatialFreqAchTest
from sklearn.metrics import root_mean_squared_error


# First we will start with the contrast detection and masking tests

input_folder = 'results_alignment_score'
output_folder = 'alignment_scores'

for in_folder in os.listdir(input_folder):
    for in_in_folder in os.listdir(os.path.join(input_folder, in_folder)):
        print(in_folder, in_in_folder)
        metrics = []
        scores = []
        for metric_file in os.listdir(os.path.join(input_folder, in_folder, in_in_folder)):
            try:
                metric_name = metric_file.replace('.csv', '')

                filepath = os.path.join(input_folder, in_folder, in_in_folder, metric_file)
                df = pd.read_csv(filepath)

                column_names = df.columns.values.tolist()

                multipliers = df[column_names[1]].to_numpy()
                x_variable = df[column_names[2]].to_numpy()
                predictions = df['Prediction'].to_numpy()

                score = spearmanr(multipliers, predictions, nan_policy='omit').statistic

                metrics.append(metric_name)
                scores.append(score)

            except:
                print('failed on ' + metric_name)

        df = pd.DataFrame({
            'Metric': metrics,
            'Score': scores
        })

        output = os.path.join(output_folder, in_folder)
        if not os.path.exists(output):
            os.mkdir(output)

        output_file = os.path.join(output, in_in_folder+'.csv')
        df.to_csv(output_file, index=False)

# Contrast Matching across color directions tests

input_folder = os.path.join('results', 'contrast_matching', 'color_directions')
print('contrast_matching', 'color_directions')

metrics = []
scores = []
for metric_file in os.listdir(input_folder):
    try:
        metric_name = metric_file.replace('.csv', '')
        filepath = os.path.join(input_folder, metric_file)
        df = pd.read_csv(filepath)

        contrasts = df['Contrast']
        predictions = df['Prediction']
        score = spearmanr(contrasts, predictions, nan_policy='omit').statistic

        metrics.append(metric_name)
        scores.append(score)

    except:
        print('failed on ' + metric_name)

df = pd.DataFrame({
    'Metric': metrics,
    'Score': scores
})

output = os.path.join(output_folder, 'contrast_matching')
if not os.path.exists(output):
    os.mkdir(output)

output_file = os.path.join(output, 'color_directions.csv')
df.to_csv(output_file, index=False)

# Contrast Matching Across spatial frequencies

input_folder = os.path.join('results', 'contrast_matching', 'spatial_frequencies')
print('contrast_matching', 'spatial_frequencies')

# Get gt
T = ContrastConstancySpatialFreqAchTest()
ref_contrasts, spatial_frequencies, gt_contrasts = T.get_gt_results()

metrics = []
scores = []
for metric_file in os.listdir(input_folder):
    try:
        metric_name = metric_file.replace('.csv', '')
        filepath = os.path.join(input_folder, metric_file)
        df = pd.read_csv(filepath)
        predictions = df['Prediction'].to_numpy().reshape(T.size())
        error = root_mean_squared_error(np.log10(np.array(gt_contrasts)), np.log10(predictions))

        metrics.append(metric_name)
        scores.append(error)
    except:
        print('failed on ' + metric_name)

df = pd.DataFrame({
    'Metric': metrics,
    'Score': scores
})

output = os.path.join(output_folder, 'contrast_matching')
if not os.path.exists(output):
    os.mkdir(output)

output_file = os.path.join(output, 'spatial_frequencies.csv')
df.to_csv(output_file, index=False)
