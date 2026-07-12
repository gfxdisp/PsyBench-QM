import numpy as np
import pandas as pd
import os
from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from metrics import *

tab_caption = 'First attempt of the table'
tab_label = 'results'

latex_text = '\\begin{table}[]\n\centering\n\caption{'+tab_caption+'}\n\label{tab:'+tab_label+'}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{|ll'

tests_cat = [r'\begin{tabular}[c]{@{}c@{}}Contrast \\ Detection\end{tabular}', 'Contrast Masking', r'\begin{tabular}[c]{@{}c@{}}Flicker \\ Detection\end{tabular}', 'Contrast Matching']

tests = [[GaborPatchSpatialFreqAchTest],
         [ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest],
         [DiskTemporalFreqAchTest],
         [ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest]]

higher_better_scale = [[True], [True, True], [True], [False, False] ]

metrics = [[None], 
           [psnry_metric, ssim_metric, ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, dss_metric, nlpd_metric],
           [fsim_metric, vsi_metric, mdsi_metric, haarpsi_metric, scielab_metric, ldr_flip_metric],
           [de2000, hyab, ictcp, msswd_metric], 
           [wadiqam_metric, lpips_metric, lpips_vgg_metric, stlpips_metric, dists_metric, ahiq_metric, topiq_metric], 
           [vmaf_met, speedqa_metric, funque_met],
           [mad_metric, hdrvdp3_metric, fvvdp_met, cvvdp_met, cvvdp_ml_saliency_met, cvvdp_ml_transformer_met]]

citations = [[None], [None, 'wang2004image', 'wang2003multiscale', 'xue2013gradient', 'zhang2017gradient', 'sheikh2006image' ,'balanov2015image', 'laparra2016perceptual'],
             ['zhang2011fsim', 'zhang2014vsi', 'nafchi2016mean', 'reisenhofer2018haar', 'zhang1996spatial', 'andersson2020flip'],
             ['CIE_2018', 'abasi2020distance', 'ITU-R_2124', 'he2024ms-swd'],
             ['bosse2017deep', 'zhang2018perceptual', 'zhang2018perceptual', 'ghildyal2022shift', 'ding2020image', 'lao2022attentions', 'chen2024topiq'],
             ['li2018vmaf', 'bampis2017speed', 'venkataramanan2022funque'],
             ['larson2010most', 'mantiuk2023hdr', 'mantiuk2021fovvideovdp', 'cvvdp', 'hammou2025colorvideovdp', 'hammou2025colorvideovdp']]

categories = [None, 'Traditional metrics', r'Traditional\\color metrics', r'Color\\difference\\measures', r'Deep-learning\\based metrics', r'Video\\metrics', r'Visual\\difference\\predictors']

# Centering
for i in range(len(tests_cat)):
    latex_text += '|'
    for j in range(len(tests[i])):
        latex_text += 'c'
latex_text += '|}'

# Header - First row
latex_text += '\n\hline\n'
#latex_text += ' & \multirow{2}{*}{Metric}'
latex_text += '\multicolumn{2}{|l|}{\multirow{2}{*}{Metric}}'
clines = []
start = 2
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

if start is not None and cline+start > start:
    clines.append([start, cline+start])

# cline information
latex_text += ' \\\ \n'
for cline in clines:
    latex_text += ' \cline{'+str(cline[0]+1)+'-'+str(cline[1])+'}'

# Results
row_num = 1
for cat_metric, cat_citation, cat in zip(metrics, citations, categories):
    for index, (metric, citation) in enumerate(zip(cat_metric, cat_citation)):
        if metric is not None:
            if index == 0:
                latex_text += ' \\\ \n\hline \hline\n'
                latex_text += '\multicolumn{1}{|l|}{\multirow{'+str(len(cat_metric))+r'}{*}{ \rotatebox[origin=c]{90}{ \begin{tabular}[c]{@{}c@{}}' + cat + ' \end{tabular}}}} &'
            else:
                latex_text += ' \\\ \n\cline{2-8}'
                latex_text += ' \multicolumn{1}{|l|}{} &'

            metric_name = metric.name()
            if metric_name == 'VMAF_v0.6.1':
                latex_text += 'VMAF v0.6.1'
            else:
                latex_text += metric_name
            if citation is not None:
                latex_text += r' \cite{'+citation+'}'
        
        else:
            latex_text += ' &'

        for i in range(len(tests_cat)):
            for test, scale in zip(tests[i], higher_better_scale[i]):
                T = test()
                if len(T.latex_name().split(' - ')[-1].split(' ')) == 3:
                    test_name = ' '.join(T.latex_name().split(' - ')[-1].split(' ')[:2]) + r'\\' + T.latex_name().split(' - ')[-1].split(' ')[-1]
                else:
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
                                bck_color = 'fff8d6'
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

        row_num += 1


latex_text += '\\\ \n\hline\n'
latex_text += '\end{tabular}%\n}\n\end{table}'

print(latex_text)




