import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import os 


input_folder = 'results_alignment_score'
output_folder = 'alignment_scores'

for in_folder in os.listdir(input_folder):
    if in_folder == 'contrast_masking':
        stop = True
    for in_in_folder in os.listdir(os.path.join(input_folder, in_folder)):
        print(in_folder, in_in_folder)
        metrics = []
        scores = []
        for metric_file in os.listdir(os.path.join(input_folder, in_folder, in_in_folder)):
            try:
                filepath = os.path.join(input_folder, in_folder, in_in_folder, metric_file)
                df = pd.read_csv(filepath)

                column_names = df.columns.values.tolist()

                multipliers = df[column_names[1]].to_numpy()
                x_variable = df[column_names[2]].to_numpy()
                predictions = df['Prediction'].to_numpy()

                metric_name = metric_file.replace('.csv', '')
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
