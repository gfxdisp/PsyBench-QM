import pandas as pd
import logging
from tqdm import trange

from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.tests_utils import get_all_tests
from metrics.piq_metrics import *
from metrics.pyiqa_metrics import *

if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    tests = get_all_tests()

    #tests = [GaborPatchSpatialFreqAchTest, GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest]
    tests = [ContrastMakingAchTest]

    output_folder = 'results'

    metrics = [dss_metric, fsim_metric, gmsd_metric, haarpsi_metric, iw_ssim_metric, mdsi_metric, ms_gmsd_metric, ms_ssim_metric, ssim_metric, vif_metric, vsi_metric, mad_metric, psnr_metric]
    
    for T in tests:
        Test = T()
        test_name = Test.short_name()

        logging.info(f"Processing {test_name} - with {len(Test)} conditions")

        full_output_folder = os.path.join(output_folder, Test.get_preview_folder())

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        scores = np.zeros((len(Test), len(metrics)))
        metric_names = []

        for i in trange(len(Test)):
            # Get condition
            vid_source, display_photometry, display_geometry = Test.get_condition(i)

            condition_scores = []

            for j, M in enumerate(metrics):

                try:
                    Metric = M()
                    metric_name = Metric.short_name()

                    if i==0:
                        metric_names.append(metric_name)

                    Metric.set_display_model(display_photometry=display_photometry, display_geometry=display_geometry)

                    with torch.no_grad():  # Do not accumulate gradients
                        score = Metric.predict_video_source(vid_source)

                    try:
                        if len(score) == 2:
                            score, stats = score
                    except:
                        pass

                    scores[i, j] = score

                    

                except:
                    logging.error(f'Failed on test {test_name}, condition {i}, and the {metric_name} metric')
                    scores[i, j] = np.nan
                    pass

            # Maybe update the progress? with the some terminal messages?
            # Like showing which test is happening
            # And then a bar of progress, until it gets to 100%

        # Here, we need to take the header of the class, which will give us information of what does the metric provide to us
        # Then, we want to get these information

        header = Test.get_row_header()
        conditions = Test.get_rows_conditions()

        df = pd.DataFrame()
        df.index.name = 'Index'
        for i in range(len(header)):
            df[header[i]] = conditions[i].cpu().numpy()


        for j in range(len(metrics)):
            df['Prediction'] = scores[:, j]

            filename = os.path.join(full_output_folder, metric_names[j]+'.csv')
            df.to_csv(filename)


