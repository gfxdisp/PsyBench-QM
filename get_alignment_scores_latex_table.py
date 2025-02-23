import numpy as np
import pandas as pd
import os
from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from metrics import *

tab_caption = 'First attempt of the table'
tab_label = 'results'

latex_text = '\\begin{table*}[]\n\centering\n\caption{'+tab_caption+'}\n\label{tab:'+tab_label+'}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{|l'

tests = [GaborPatchSpatialFreqAchTest, GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest, GaborPatchSpatialFreqAchTransientTest, 
         GaborPatchLuminanceAchTest, GaborPatchSizeAchTest, 
         ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest, 
         DiskTemporalFreqAchTest,
         ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest]

metrics = [None, psnry_metric, ssim_metric, ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, fsim_metric, vsi_metric, mdsi_metric, dss_metric, haarpsi_metric, mad_metric, nlpd_metric, wadiqam_metric, lpips_metric, lpips_vgg_metric, dists_metric, ahiq_metric, topiq_metric, ldr_flip_metric, vmaf_met, funque_met, fvvdp_met, cvvdp_met]

for i in range(len(tests)):
    latex_text += 'c'

latex_text += '|}'
for metric in metrics:
    if metric is None:
        latex_text += '\n\hline\n'
        latex_text += 'Metric'
    else: 
        latex_text += ' \\\ \n\hline\n'
        metric_name = metric.name()
        if metric_name == 'VMAF_v0.6.1':
            latex_text += 'VMAF v0.6.1'
        else:
            latex_text += metric_name
    for test in tests:
        T = test()
        test_name = T.latex_name().replace(' - ', r'\\')
        if metric is None:
            latex_text += ' & \\begin{tabular}[c]{@{}c@{}} '+test_name+ '\end{tabular}'
        else:
            scores_folder = os.path.join('alignment_scores', T.get_preview_folder()+'.csv')
            try:
                df = pd.read_csv(scores_folder)
                score = df[df['Metric'] == metric_name]['Score'].to_numpy()[0]
                latex_text += ' & '+ str(np.abs(np.round(score, 3))) 
            except:
                latex_text += ' & -'

latex_text += '\\\ \n\hline\n'
latex_text += '\end{tabular}%\n}\n\end{table*}'

print(latex_text)




