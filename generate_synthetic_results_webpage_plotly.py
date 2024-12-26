import pandas as pd
import logging
from tqdm import trange
import shutil

from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.tests_utils import get_all_tests
from metrics import *

from test_reports_utils.html_report import html_report
from test_reports_utils.js_report import js_report


if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)


    tests = [GaborPatchSpatialFreqAchTest, GaborPatchSizeAchTest, ContrastMakingAchTest]

    metrics = [cvvdp_met, fvvdp_met, funque_met, vmaf_met, psnry_metric, ssim_metric, ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, fsim_metric, vsi_metric, mdsi_metric, dss_metric, haarpsi_metric, mad_metric, nlpd_metric, wadiqam_metric, lpips_metric, lpips_vgg_metric, dists_metric, ahiq_metric, topiq_metric, pieapp_metric]

    tests = [GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest]

    metrics = [cvvdp_met, fsim_metric, vsi_metric, mdsi_metric, haarpsi_metric, wadiqam_metric, lpips_metric, lpips_vgg_metric, dists_metric, ahiq_metric, topiq_metric, pieapp_metric]

    tests = [GaborPatchLuminancePU21AchTest, GaborPatchLuminancePQAchTest, GaborPatchLuminanceHLGAchTest]

    metrics = [funque_met, vmaf_met, psnry_metric, ssim_metric, ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, fsim_metric, vsi_metric, mdsi_metric, dss_metric, haarpsi_metric, mad_metric, nlpd_metric, wadiqam_metric, lpips_metric, lpips_vgg_metric, dists_metric, ahiq_metric, topiq_metric, pieapp_metric]

    tests = [GaborPatchLuminanceAchTest]

    metrics = [cvvdp_met, fvvdp_met]


    input_folder = 'results'
    output_folder = 'synthetic_test_webpage_plotly'

    generate_stimuli = False
    generate_metrics_results = True

    

    # Create the directory and its associated html file!

    if not os.path.exists(output_folder):
        os.makedirs(os.path.join(output_folder, 'html_files'))
        os.makedirs(os.path.join(output_folder, 'previews'))
        os.makedirs(os.path.join(output_folder, 'plots'))
        os.makedirs(os.path.join(output_folder, 'css_files'))

    shutil.copyfile(os.path.join('test_reports_utils', 'style.css'), os.path.join(output_folder, 'css_files', 'style.css'))

    header_file = os.path.join('test_reports_utils', "header_plotly.html")
    hr_main = html_report(os.path.join(output_folder, 'index.html'), header_file=header_file )

    js_main = js_report(os.path.join(output_folder, 'js_files', 'script.js'))

    for T in tests:
        Test = T()
        test_name = Test.short_name()
        [test_rows, test_columns] = Test.size()
        [x_variable_name, y_variable_name] = Test.get_row_header()
        [x_variable, y_variable] = Test.get_rows_conditions()

        logging.info(f"Generating results for {test_name}")

        preview_folder = os.path.join(output_folder, 'previews', Test.get_preview_folder())
        plot_folder = os.path.join(output_folder, 'plots', Test.get_preview_folder())
        results_folder = os.path.join('results', Test.get_preview_folder())

        if not os.path.exists(preview_folder):
            os.makedirs(preview_folder)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        # Put a header for the test, and link it to the webpage of the previews

        hr_main.insert_separator()
        hr_main.insert_header(test_name)

        # Here we want to create an html file for each test!

        test_html_filename = test_name.lower().replace(' ', '_')+'.html'
        hr_test = html_report(os.path.join(output_folder, test_html_filename), header_file=header_file)

        hr_main.insert_text('<p>See the test image <a href={link}>gallery</a>.</p>'.format(link=test_html_filename))

        hr_test.beg_table()

        # Now, we want to go through each condition, get both test and ref figures, save them, and put them in the webpage!
        for i in np.linspace(0, test_rows-1, 5).astype(int):
            hr_test.beg_table_row()
            for j in np.linspace(0, test_columns-1, 5).astype(int):
                hr_test.beg_table_cell()

                condition_index = i*test_columns+j

                test_filename = os.path.join(preview_folder, 'test_' + str(condition_index))
                ref_filename = os.path.join(preview_folder, 'ref_' + str(condition_index))

                if condition_index == 63:
                    stop_me_here = True

                if Test.get_condition_file_format() == 'image':
                    test_filename += '.png'
                    ref_filename += '.png'
                    if generate_stimuli:
                        Test.save_as_png(condition_index, test_filename, ref_filename)

                    # Add them in the webpage!

                    test_filename = os.path.normpath(test_filename).split(os.path.sep)[1:]
                    test_filename = os.path.join(*test_filename)

                    ref_filename = os.path.normpath(ref_filename).split(os.path.sep)[1:]
                    ref_filename = os.path.join(*ref_filename)

                    hr_test.beg_div(class_tag='cell-content')
                    hr_test.insert_image(test_filename, mousedown_image=ref_filename, class_tag="toggle-image")
                    hr_test.insert_text(f"{x_variable_name} = {x_variable[condition_index]:.2g}, {y_variable_name} = {y_variable[condition_index]:.2g}")
                    hr_test.close_div()

                elif Test.get_condition_file_format() == 'video':
                    test_filename += '.mp4'
                    ref_filename += '.mp4'
                    if generate_stimuli:
                        Test.save_as_video(condition_index, test_filename, ref_filename)

                    # Add them in the webpage!

                    test_filename = os.path.normpath(test_filename).split(os.path.sep)[1:]
                    test_filename = os.path.join(*test_filename)

                    ref_filename = os.path.normpath(ref_filename).split(os.path.sep)[1:]
                    ref_filename = os.path.join(*ref_filename)

                    hr_test.beg_div(class_tag='cell-content')
                    hr_test.insert_video(test_filename, mousedown_video=ref_filename, class_tag="toggle-video")
                    hr_test.insert_text(f"{x_variable_name} = {x_variable[condition_index]:.2g}, {y_variable_name} = {y_variable[condition_index]:.2g}")
                    hr_test.close_div()
                    

                hr_test.end_table_cell()
            hr_test.end_table_row()
        hr_test.end_table()

        hr_main.beg_div(class_tag='image-container')

        for j, M in enumerate(metrics):

            try:

                metric_name = M.name()

                metric_id = test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')+'_'+metric_name.lower().replace(' ', '_')

                metric_predictions_file = os.path.join(results_folder, metric_name+'.csv')

                metric_predictions_df = pd.read_csv(metric_predictions_file)

                predictions = metric_predictions_df['Prediction'].to_numpy()

                if generate_metrics_results:
                    js_command = Test.plotly(predictions, reverse_color_order=M.is_lower_better(), title=metric_name+' Metric', fig_id=metric_id)

                # Similarly to the main html file we need to create a JS file!

                js_main.println(js_command)

                hr_main.beg_div(id_tag = metric_id)
                hr_main.close_div()

            except:
                logging.error(f'Failed on test {test_name} and the {metric_name} metric')

        hr_main.close_div()

        hr_test.close()

    hr_main.close()
    js_main.close()





