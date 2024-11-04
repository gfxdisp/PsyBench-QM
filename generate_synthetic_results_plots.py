import pandas as pd
import logging
from tqdm import trange

from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.tests_utils import get_all_tests
from metrics import *

if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    tests = [GaborPatchSpatialFreqAchTest, GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest]

    input_folder = 'results'
    output_folder = 'plots'

    metrics = [funque_met, psnry_metric, fsim_metric, cvvdp_met, vmaf_met, gmsd_metric]

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
                metric_name = M.name()

                metric_predictions_file = os.path.join(full_input_folder, metric_name+'.csv')

                metric_predictions_df = pd.read_csv(metric_predictions_file)

                predictions = metric_predictions_df['Prediction'].to_numpy()

                output_filename = os.path.join(full_output_folder, metric_name+'.png')

                Test.plot(predictions, reverse_color_order=M.is_lower_better(), output_filename=output_filename, title=metric_name+' Metric')

            except:
                logging.error(f'Failed on test {test_name} and the {metric_name} metric')




