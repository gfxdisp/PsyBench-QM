from metrics import cvvdp_met
from pyst.ContrastMatching import *
from pyst.ContrastMasking import *
from pyst.GaborPatch import *
import matplotlib.pyplot as plt
import numpy as np


""" First thing we want to generate the necessary images for the grid
which tests to consider:
* Contrast Detection - Spatial Freq 
* Contrast Masking - Phase Coherent
* Contrast Matching - Both tests 
"""

# Contrast Detection
T = GaborPatchSpatialFreqAchTest(width=512, height=512, device='cpu')
y_indices = [0, 15, 19]
x_indices = [0, 6, 17]

i = 0
for y_index in y_indices:
    for x_index in x_indices:
        index = y_index*20+x_index
        T.save_as_png(index, 'figs/detection/test_'+str(i)+'.png', 'figs/detection/ref.png')
        i += 1

# Contrast masking
T = ContrastMakingCoherentAchTest(width=512, height=512, device='cpu')
x_indices = [0, 15, 19]
y_indices = [0, 15, 19]

i = 0
for y_index in y_indices:
    for x_index in x_indices:
        index = y_index*20+x_index
        T.save_as_png(index, 'figs/masking/test_'+str(i)+'.png', 'figs/masking/ref'+str(i)+'.png')
        i += 1

# Contrast Matching - Freq
T = ContrastConstancySpatialFreqAchTest(width=512, height=512, device='cpu')
x_indices = [0, 4, 8]
y_indices = [0, 2, 6]

i = 0
for y_index in y_indices:
    for x_index in x_indices:
        index = y_index*9+x_index
        T.save_as_png(index, 'figs/matching_freq/test_'+str(i)+'.png', 'figs/matching_freq/ref'+str(i)+'.png')
        i += 1

# Contrast Matching - color
T = ContrastConstancyColorTest(width=512, height=512, device='cpu')
x_indices = [0, 1, 2]
y_indices = [0, 6, 9]

i = 0
for y_index in y_indices:
    for x_index in x_indices:
        index = y_index*3+x_index
        T.save_as_png(index, 'figs/matching_color/test_'+str(i)+'.png', 'figs/matching_color/ref'+str(i)+'.png')
        i += 1

""" Generate results for each test """
metric = cvvdp_met
metric_name = metric.name()

for test in [GaborPatchSpatialFreqAchTest, ContrastMakingCoherentAchTest, ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest]:

    T = test(device='cpu')
    test_name = T.short_name()

    results_folder = os.path.join('results', T.get_preview_folder())

    alignment_scores_filepath = os.path.join('alignment_scores', T.get_preview_folder()+'.csv')
    alignment_scores_df = pd.read_csv(alignment_scores_filepath)

    metric_predictions_file = os.path.join(results_folder, metric_name + '.csv')
    metric_predictions_df = pd.read_csv(metric_predictions_file)
    predictions = metric_predictions_df['Prediction'].to_numpy()

    try:
        alignment_score = alignment_scores_df[alignment_scores_df['Metric'] == metric_name]['Score'].to_numpy()[0]
        if test_name.split(' - ')[0] == 'Contrast Matching':
            title = metric_name + ' [RMSE=' + str(round(np.abs(alignment_score), 3)) + ']'
        else:
            title = metric_name + ' [AS=' + str(round(np.abs(alignment_score), 3)) + ']'
    except:
        title = metric_name + ' Metric'

    filename = 'figs/'+metric_name+'_'+test_name.replace(' - ', '_')+'.png'

    T.plot(predictions, reverse_color_order=metric.is_lower_better(), title=title, output_filename=filename,is_first_column=True, fontsize=18)






