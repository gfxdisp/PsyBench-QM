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

tests_cat = ['Contrast Detection', 'Contrast Masking', r'\begin{tabular}[c]{@{}c@{}}Flicker \\ Detection\end{tabular}', 'Contrast Matching']

tests = [[GaborPatchSpatialFreqAchTest, GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest, GaborPatchSpatialFreqAchTransientTest,
         GaborPatchLuminanceAchTest, GaborPatchSizeAchTest],
         [ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest],
         [DiskTemporalFreqAchTest],
         [ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest]]

higher_better_scale = [[True, True, True, True, True, True], [True, True], [True], [False, True] ]

metrics = [None, psnry_metric, ssim_metric, ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, fsim_metric, vsi_metric, mdsi_metric, dss_metric, haarpsi_metric, mad_metric, nlpd_metric, wadiqam_metric, lpips_metric, lpips_vgg_metric, dists_metric, ahiq_metric, topiq_metric, ldr_flip_metric, vmaf_met, funque_met, fvvdp_met, cvvdp_met]

# Centering
for i in range(len(tests_cat)):
    latex_text += '|'
    for j in range(len(tests[i])):
        latex_text += 'c'
latex_text += '|}'

# Header - First row
latex_text += '\n\hline\n'
latex_text += '\multirow{2}{*}{Metric}'
clines = []
start = 1
cline = 0
for i in range(len(tests_cat)):
    if len(tests[i]) == 1:
        latex_text += ' & \multirow{2}{*}{'+tests_cat[i]+'}'
        clines.append([start, cline+start, i])
        start = None
        cline = 0
    else:
        if start is None:
            start = clines[-1][1] + i - clines[-1][2]
        latex_text += ' & \multicolumn{'+str(len(tests[i]))+'}{c|}{'+tests_cat[i]+'}'
        cline += len(tests[i])

if start is not None:
    clines.append([start, cline+start])

# cline information
latex_text += ' \\\ \n'
for cline in clines:
    latex_text += ' \cline{'+str(cline[0]+1)+'-'+str(cline[1])+'}'

# Results
for metric in metrics:
    if metric is not None:
        latex_text += ' \\\ \n\hline\n'
        metric_name = metric.name()
        if metric_name == 'VMAF_v0.6.1':
            latex_text += 'VMAF v0.6.1'
        else:
            latex_text += metric_name

    for i in range(len(tests_cat)):
        for test, scale in zip(tests[i], higher_better_scale[i]):
            T = test()
            test_name = r'\\'.join(T.latex_name().split(' - ')[2:])
            #test_name = T.latex_name().replace(' - ', r'\\')
            if metric is None:
                if len(tests[i]) > 1:
                    latex_text += ' & \multicolumn{1}{c|}{\\begin{tabular}[c]{@{}c@{}} '+test_name+ '\end{tabular}}'
                else:
                    latex_text += ' & '
            else:
                scores_folder = os.path.join('alignment_scores', T.get_preview_folder()+'.csv')
                try:
                    df = pd.read_csv(scores_folder)
                    df.fillna(0, inplace=True)
                    scores = np.abs(df['Score'].to_numpy())
                    """ Get the different quantiles """
                    score = np.abs(df[df['Metric'] == metric_name]['Score'].to_numpy()[0])

                    if scale:
                        if score >= np.quantile(scores, 0.75):
                            bck_color = 'cbe1cd'
                        elif score >= np.quantile(scores, 0.5):
                            bck_color = 'fff8d6'
                        elif score >= np.quantile(scores, 0.25):
                            bck_color = 'ffd8b8'
                        else:
                            bck_color = 'fdddd8'

                    else:
                        if score <= np.quantile(scores, 0.25):
                            bck_color = 'cbe1cd'
                        elif score <= np.quantile(scores, 0.5):
                            bck_color = 'f5edc0'
                        elif score <= np.quantile(scores, 0.75):
                            bck_color = 'ffd8b8'
                        else:
                            bck_color = 'fdddd8'

                    latex_text += ' & \multicolumn{1}{c|}{\cellcolor[HTML]{'+bck_color+'} '+ str(np.round(score, 3))
                    if score == 0:
                        latex_text += ' (*)'

                except:
                    latex_text += ' & \multicolumn{1}{c|}{\cellcolor[HTML]{f1f1f1} -'

                latex_text += '}'



latex_text += '\\\ \n\hline\n'
latex_text += '\end{tabular}%\n}\n\end{table*}'

print(latex_text)




