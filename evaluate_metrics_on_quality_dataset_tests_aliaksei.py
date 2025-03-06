import pandas as pd
import logging
from tqdm import trange
import os
from pyst.TID2013 import TID2013
from pyst.LosslessDataset import LosslessDataset
from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from metrics import *
import numpy as np
import sys
import torch
from scipy.stats import spearmanr

if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    tests = [TID2013]
    tests = [LosslessDataset]
    metrics = [cvvdp_met, fvvdp_met, funque_met, ldr_flip_metric, vmaf_met, psnry_metric, ssim_metric,
               ms_ssim_metric, vifp_metric, fsim_metric, vsi_metric, mdsi_metric,
               dss_metric, haarpsi_metric, mad_metric, nlpd_metric, wadiqam_metric, lpips_metric, lpips_vgg_metric,
               dists_metric, ahiq_metric, topiq_metric, ictcp, stlpips_metric,
               deepdc_metric]

    # Get figures

    folder = 'results_quality_datasets'

    # for T in tests:
    #     test = T()
    #     test_name = test.short_name()
    #     in_folder = os.path.join(folder, test.get_preview_folder())
    #
    #     #for metric_file in os.listdir(in_folder):
    #         # metric_name = metric_file.replace('.csv', '')
    #     for metric in metrics:
    #         metric_name = metric.name()
    #         metric_file=metric_name+'.csv'
    #         min_val, max_val = metric.predictions_range()
    #         df = pd.read_csv(os.path.join(in_folder, metric_file))
    #         predictions = df['Prediction'].to_numpy()
    #         #test.optimize(predictions, min_val, max_val)
    #         test.plot(predictions, min_val, max_val, metric_name)

    # Get correlation with all other tests!

    comparable_tests = [GaborPatchSpatialFreqAchTest, GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest,
                        GaborPatchLuminanceAchTest, GaborPatchSizeAchTest,
                        ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest,
                        ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest]

    for T in tests:
        test = T()
        in_folder = os.path.join(folder, test.get_preview_folder())
        for comparable_test in comparable_tests:
            errors = []
            rmses = []
            metric_names = []
            test2 = comparable_test()
            df_rmse = pd.read_csv(os.path.join('alignment_scores', test2.get_preview_folder()+'.csv'))
            for metric in metrics:
                metric_name = metric.name()
                min_val, max_val = metric.predictions_range()
                metric_file = metric_name+'.csv'
                df = pd.read_csv(os.path.join(in_folder, metric_file))
                predictions = df['Prediction'].to_numpy()

                try:
                    rmses.append(np.abs(df_rmse[df_rmse['Metric'] == metric_name]['Score'].to_numpy()[0]))#
                    errors.append(np.abs(test.error(predictions, min_val, max_val)))
                    metric_names.append(metric_name)
                except:
                    #rmses.append(np.nan)
                    pass

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(errors, rmses)
            for val1, val2, val3 in zip(errors, rmses, metric_names):
                plt.text(val1, val2, val3)
            plt.xlabel('RMSE on VLIC')
            plt.ylabel('Alignment Score')
            plt.title(test2.short_name()+', Corr='+str(np.round(spearmanr(errors, rmses, nan_policy='omit').statistic, 3)))
            fig.savefig(os.path.join('figs/aliaksei_3', test2.short_name()+'.png'))

            print(test2.short_name())
            print(spearmanr(errors, rmses, nan_policy='omit'))



