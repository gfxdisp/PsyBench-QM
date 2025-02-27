import numpy as np
import pandas as pd
import os
from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from metrics import *


tests = [GaborPatchSpatialFreqAchTest, GaborPatchSpatialFreqRGTest, GaborPatchLuminanceAchTest, GaborPatchSizeAchTest,
         ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest,
         DiskTemporalFreqAchTest,
         ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest]

# We will have to define different metrics for different tests
metrics =[[psnry_metric, ssim_metric, dists_metric, ldr_flip_metric, cvvdp_met],
          [fsim_metric, lpips_metric, dists_metric, ldr_flip_metric, cvvdp_met],
          [psnry_metric, funque_met, lpips_metric, ldr_flip_metric, hdrvdp3_metric],
          [psnry_metric, mad_metric, lpips_metric, topiq_metric, cvvdp_met],
          [psnry_metric, vmaf_met, lpips_metric, fvvdp_met, cvvdp_met],
          [psnry_metric, lpips_vgg_metric, dists_metric, hdrvdp3_metric, cvvdp_met],
          [vmaf_met, speedqa_metric, funque_met, fvvdp_met, cvvdp_met],
          [psnry_metric, dss_metric, dists_metric, ldr_flip_metric, cvvdp_met],
          [fsim_metric, de2000, dists_metric, ldr_flip_metric, cvvdp_met]]



output_path = 'figs'

# Create the main figure
fig, axes = plt.subplots(len(tests), len(metrics[0]), figsize=((23, 3*len(tests))))

for i, test in enumerate(tests):
    T = test()
    test_name = T.short_name()
    results_folder = os.path.join('results', T.get_preview_folder())
    try:
        alignment_scores_filepath = os.path.join('alignment_scores', T.get_preview_folder()+'.csv')
        alignment_scores_df = pd.read_csv(alignment_scores_filepath)
    except:
        logging.error(f'No Alignment scores for {test_name}')

    for j, metric in enumerate(metrics[i]):
        metric_name = metric.name()
        # get predictions

        metric_predictions_file = os.path.join(results_folder, metric_name+'.csv')
        metric_predictions_df = pd.read_csv(metric_predictions_file)

        predictions = metric_predictions_df['Prediction'].to_numpy()

        # Get Alignment Score
        try:
            alignment_score = alignment_scores_df[alignment_scores_df['Metric'] == metric_name]['Score'].to_numpy()[0]
            if test_name.split(' - ')[0] == 'Contrast Matching':
                title = metric_name + ' Metric\nRMSE = ' + str(round(np.abs(alignment_score), 3))
            else:
                title = metric_name + ' Metric\nAlignment Score = '+str(round(np.abs(alignment_score), 3))
        except:
            title = metric_name + ' Metric'

        # call the plot function and save it for now
        T.plot(predictions, reverse_color_order=metric.is_lower_better(), title=title, axis=axes[i, j], is_first_column=j==0)
    
    #axes[i, 0].set_ylabel('Test1', fontsize=16, fontweight='bold', labelpad=30)
    if test_name.split(' - ')[0] == 'Contrast Matching':
        axes[i, 0].text(-0.38, 0.5, f"({chr(97 + i)})", fontweight='bold', fontsize=15, rotation=90, ha='center', va='center', transform=axes[i, 0].transAxes)
    else:
        axes[i, 0].text(-0.48, 0.5, f"({chr(97 + i)})", fontweight='bold', fontsize=15, rotation=90, ha='center', va='center', transform=axes[i, 0].transAxes)


# Save figure    
filename = os.path.join(output_path, 'test.pdf')

plt.subplots_adjust(left=0)
plt.tight_layout()
fig.savefig(filename)
plt.close(fig)