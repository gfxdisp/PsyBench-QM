from metrics import *
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import os
from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *



metrics = [psnry_metric, ssim_metric, ms_ssim_metric, vifp_metric, fsim_metric, vsi_metric, mad_metric,
           lpips_metric, lpips_vgg_metric, dists_metric]

correlations = [0.2547, 0.3399, 0.4863, 0.4334, 0.4678, 0.4500, 0.6077, 0.6275, 0.5914, 0.6742]

# I will remove FSIM, VSI, WADIQAM, VIFp, TOPIQ, and VMAF

comparable_tests = [GaborPatchSpatialFreqAchTest, GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest, GaborPatchSpatialFreqAchTransientTest,
                    GaborPatchLuminanceAchTest, GaborPatchSizeAchTest,
                    ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest,
                    DiskTemporalFreqAchTest,
                    ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest]

for T in comparable_tests:
    test = T(device='cpu')
    df1 = pd.read_csv(os.path.join('alignment_scores', test.get_preview_folder()+'.csv'))
    corr1 = []
    corr2 = []
    metric_names = []
    for metric, corr in zip(metrics, correlations):
        metric_name = metric.name()
        df2 = pd.read_csv(os.path.join('quality_dataset_tests', 'xr_david_results', 'XR-DAVID_'+metric_name+'.csv'))
        try:
            corr1.append(df1[df1['Metric'] == metric_name]['Score'].to_numpy()[0])
            corr2.append(corr)
            metric_names.append(metric_name)
        except:
            pass


    fig = plt.figure(figsize=(8, 8))
    plt.scatter(np.abs(corr2), np.abs(corr1))
    for val1, val2, metric_name in zip(corr1, corr2, metric_names):
        plt.text(np.abs(val2), np.abs(val1), metric_name)
    plt.xlabel('Correlation PIPAL')
    plt.ylabel('Alignment Score')
    plt.title(test.short_name()+', Corr = '+str(np.round(spearmanr(np.abs(corr1), np.abs(corr2)).statistic, 3)))
    fig.savefig(os.path.join('figs/pipal', test.short_name()+'.png'), bbox_inches='tight')

    print(test.short_name())
    print(spearmanr(np.abs(corr1), np.abs(corr2)))
