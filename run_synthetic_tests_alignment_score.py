import pandas as pd
import logging
from tqdm import trange

from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from metrics import *

import sys

if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    """ We will need at some point to add arguments, you simply call this with all tests and metrics for those tests, so we can parallalize it easily or run it with ssh"""

    test_index = int(sys.argv[1])

    if test_index == 0:
        tests = [GaborPatchSpatialFreqAchTest, GaborPatchLuminanceAchTest, GaborPatchSizeAchTest, ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest]
        metrics = [cvvdp_met, fvvdp_met, funque_met, ldr_flip_metric, vmaf_met, psnry_metric, ssim_metric,
                   ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, fsim_metric, vsi_metric, mdsi_metric,
                   dss_metric, haarpsi_metric, mad_metric, nlpd_metric, wadiqam_metric, lpips_metric, lpips_vgg_metric,
                   dists_metric, ahiq_metric, topiq_metric, hyab, de2000, ictcp, msswd_metric, stlpips_metric, deepdc_metric]
        # metrics = [scielab_metric, hdrvdp3_metric, speedqa_metric]
        tests = [GaborPatchSpatialFreqAchTest]
        metrics = [hdrvdp3_metric]
        

    elif test_index == 1:
        tests = [GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest]
        metrics = [cvvdp_met, ldr_flip_metric, fsim_metric, vsi_metric, mdsi_metric, haarpsi_metric, wadiqam_metric,
                   lpips_metric, lpips_vgg_metric, dists_metric, ahiq_metric, topiq_metric, hyab, de2000, ictcp, msswd_metric, stlpips_metric, deepdc_metric]
        #metrics = [scielab_metric]

    elif test_index == 2:
        tests = [DiskTemporalFreqAchTest, GaborPatchSpatialFreqAchTransientTest]
        metrics = [cvvdp_met, fvvdp_met, vmaf_met, funque_met]
        #metrics = [speedqa_metric]
        tests = [GaborPatchSpatialFreqAchTransientTest]

    output_folder = 'results_alignment_score'

    metric_classes = []
    for j, M in enumerate(metrics):
        metric_classes.append(M())

    """ Here we are only interested in the data points that are similar to the ground truth """
    """ We will first get the ground truth of each test, depending on the ground truth, we will define the contrasts """
    """ These contrasts would be multiplied by the multipliers, and used to get the predictions at the different axes """

    for T in tests:
        Test = T(is_alignmenet_score=True)
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
                    # Metric = M()
                    Metric = metric_classes[j]
                    metric_name = M.name()

                    if i == 0:
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

            filename = os.path.join(full_output_folder, metric_names[j] + '.csv')
            df.to_csv(filename)


