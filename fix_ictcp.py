import os

folders = ['results', 'results_alignment_score']

for folder in folders:
    for sub_folder in os.listdir(folder):
        for sub_sub_folder in os.listdir(os.path.join(folder, sub_folder)):
            try:
                in_path = os.path.join(folder, sub_folder, sub_sub_folder, 'Dolby-ICTCP.csv')
                out_path = os.path.join(folder, sub_folder, sub_sub_folder, 'ICtCp.csv')
                os.system('mv '+in_path+' '+out_path)
            except:
                print(folder, sub_folder)