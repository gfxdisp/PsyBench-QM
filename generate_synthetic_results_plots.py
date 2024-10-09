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
    tests = [ContrastMakingAchTest]

    input_folder = 'results'
    output_folder = 'plots'

    metrics = [dss_metric, fsim_metric, gmsd_metric, haarpsi_metric, iw_ssim_metric, mdsi_metric, ms_gmsd_metric,
               ms_ssim_metric, ssim_metric, vif_metric, vsi_metric, mad_metric, psnr_metric]

    for T in tests:
        Test = T()
        test_name = Test.short_name()

        logging.info(f"Processing {test_name} - with {len(Test)} conditions")

        full_output_folder = os.path.join(output_folder, Test.get_preview_folder())
        full_input_folder = os.path.join(input_folder, Test.get_preview_folder())

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        for j, M in enumerate(metrics):

            try:
                Metric = M()
                metric_name = Metric.short_name()

                metric_predictions_file = os.path.join(full_input_folder, metric_name+'.csv')

                metric_predictions_df = pd.read_csv(metric_predictions_file)

                predictions = metric_predictions_df['Prediction'].to_numpy()

                output_filename = os.path.join(full_output_folder, metric_name+'.png')

                Test.plot(predictions, output_filename=output_filename, title=metric_name+' Metric')

            except:
                logging.error(f'Failed on test {test_name} and the {metric_name} metric')




