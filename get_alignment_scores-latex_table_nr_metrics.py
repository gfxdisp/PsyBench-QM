import numpy as np
import pandas as pd
import os
from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from metrics import *

tab_caption = 'First attempt of the table'
tab_label = 'results'

latex_text = '\\begin{table}[]\n\centering\n\caption{'+tab_caption+'}\n\label{tab:'+tab_label+'}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{|ll'

tests_cat = ['Contrast Detection', 'Contrast Masking', 'Contrast Matching']

tests = [[GaborPatchSpatialFreqAchTest, GaborPatchSpatialFreqRGTest, GaborPatchSizeAchTest],
         [ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest],
         [ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest]]

higher_better_scale = [[True, True, True, True], [True, True], [False, False]]

metrics = [[None], 
           [nima_metric, hyperiqa_metric, paq2piq_metric, musiq_metric, arniqa_metric, tres_metric]]

citations = [[None], [None, None, None, None, None, None]]

categories = [None, 'NR metrics']

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

if start is not None:
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
                latex_text += ' \\\ \n\cline{2-9}'
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

                        # if scale:
                        #     if score >= np.quantile(scores, 0.75):
                        #         bck_color = 'cbe1cd'
                        #     elif score >= np.quantile(scores, 0.5):
                        #         bck_color = 'fff8d6'
                        #     elif score >= np.quantile(scores, 0.25):
                        #         bck_color = 'ffd8b8'
                        #     else:
                        #         bck_color = 'fdddd8'

                        # else:
                        #     if score <= np.quantile(scores, 0.25):
                        #         bck_color = 'cbe1cd'
                        #     elif score <= np.quantile(scores, 0.5):
                        #         bck_color = 'fff8d6'
                        #     elif score <= np.quantile(scores, 0.75):
                        #         bck_color = 'ffd8b8'
                        #     else:
                        #         bck_color = 'fdddd8'
                        
                        if (scale and score > 0.6) or (not scale and score < 0.4) :
                            bck_color = 'cbe1cd'
                            latex_text += ' & \multicolumn{1}{c|}{\cellcolor[HTML]{'+bck_color+'} '+ str(np.round(score, 3))
                        else:
                            latex_text += ' & \multicolumn{1}{c|}{ '+ str(np.round(score, 3))

                        #latex_text += ' & \multicolumn{1}{c|}{\cellcolor[HTML]{'+bck_color+'} '+ str(np.round(score, 3))
                        #latex_text += ' & \multicolumn{1}{c|}{ '+ str(np.round(score, 3))
                        if score == 0:
                            latex_text += ' (*)'

                    except:
                        latex_text += ' & \multicolumn{1}{c|}{\cellcolor[HTML]{f1f1f1} -'

                    latex_text += '}'

        row_num += 1


latex_text += '\\\ \n\hline\n'
latex_text += '\end{tabular}%\n}\n\end{table}'

print(latex_text)




