from metrics import *
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import os
from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *


metrics = [psnry_metric, ssim_metric, ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, dss_metric, mad_metric, nlpd_metric,
               fsim_metric, vsi_metric, mdsi_metric, haarpsi_metric,
               de2000, hyab, ictcp, msswd_metric,
               wadiqam_metric, lpips_metric, lpips_vgg_metric, stlpips_metric, dists_metric, ahiq_metric, topiq_metric, deepdc_metric,
               vmaf_met, speedqa_metric, funque_met,
               scielab_metric, ldr_flip_metric, hdrvdp3_metric, fvvdp_met, cvvdp_met]

metrics = [psnry_metric, ssim_metric, ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, dss_metric, nlpd_metric,
               fsim_metric, vsi_metric, mdsi_metric, haarpsi_metric,
               de2000, hyab, ictcp, msswd_metric,
               wadiqam_metric, lpips_metric, lpips_vgg_metric, stlpips_metric, dists_metric, ahiq_metric, topiq_metric, deepdc_metric,
               funque_met, vmaf_met,
               ldr_flip_metric, hdrvdp3_metric, fvvdp_met, cvvdp_met]

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
    for metric in metrics:
        metric_name = metric.name()
        df2 = pd.read_csv(os.path.join('quality_dataset_tests', 'xr_david_results', 'XR-DAVID_'+metric_name+'.csv'))
        try:
            corr1.append(df1[df1['Metric'] == metric_name]['Score'].to_numpy()[0])
            corr2.append(spearmanr(df2['Q'].to_numpy(), df2['Q_subj'].to_numpy()).statistic)
            metric_names.append(metric_name)
        except:
            pass

    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(np.abs(corr2), np.abs(corr1))
    for val1, val2, metric_name in zip(corr1, corr2, metric_names):
        #plt.scatter(np.abs(val2), np.abs(val1), label=metric_name)
        plt.text(np.abs(val2), np.abs(val1), metric_name)
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Correlation XR-DAVID')
    plt.ylabel('Alignment Score')
    plt.title(test.short_name()+', Corr = '+str(np.round(spearmanr(np.abs(corr1), np.abs(corr2)).statistic, 3)))
    fig.savefig(os.path.join('figs/xr_david', test.short_name()+'.png'), bbox_inches='tight')
    
    print(test.short_name())
    print(spearmanr(np.abs(corr1), np.abs(corr2)))
