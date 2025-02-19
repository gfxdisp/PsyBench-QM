import pandas as pd
import logging
from tqdm import trange
import shutil

from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from metrics import *

from test_reports_utils.html_report import html_report
from test_reports_utils.js_report import js_report


if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)


    tests = [GaborPatchSpatialFreqAchTest, BandLimitedNoiseSpatialFreqAchTest, GaborPatchSizeAchTest, 
             ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest,
             GaborPatchSpatialFreqAchTransientTest, DiskTemporalFreqAchTest,
             GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest, BandLimitedNoiseSpatialFreqRGTest, BandLimitedNoiseSpatialFreqYVTest,
             GaborPatchLuminancePU21AchTest, GaborPatchLuminancePQAchTest, GaborPatchLuminanceHLGAchTest,
             GaborPatchLuminanceAchTest]

    metrics = [cvvdp_met, fvvdp_met, funque_met, vmaf_met, hdr_flip_metric, ldr_flip_metric, psnry_metric, ssim_metric, ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, fsim_metric, vsi_metric, mdsi_metric, dss_metric, haarpsi_metric, mad_metric, nlpd_metric, wadiqam_metric, dists_metric, ahiq_metric, topiq_metric, lpips_metric, lpips_vgg_metric]

    tests = [ContrastConstancyColorTest]

    input_folder = 'results'
    output_folder = 'synthetic_test_webpage_plotly'

    generate_stimuli = True
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

    hr_main.insert_file(os.path.join('test_reports_utils', "nav.html"))
    hr_main.beg_div(class_tag='other_content')
    hr_main.insert_text('Please select one of the tests in the sidebar to view the models performance in the test.')
    hr_main.close_div()

    for T in tests:
        Test = T()
        test_name = Test.short_name()
        [test_rows, test_columns] = Test.size()
        [x_variable_name, y_variable_name] = Test.get_row_header()
        [x_variable, y_variable] = Test.get_rows_conditions()

        logging.info(f"Generating results for {test_name}")

        preview_folder = os.path.join(output_folder, 'previews', Test.get_preview_folder())
        results_folder = os.path.join('results', Test.get_preview_folder())

        # Get alignment scores data for this test and all metrics !
        try:
            alignment_scores_filepath = os.path.join('alignment_scores', Test.get_preview_folder()+'.csv')
            alignment_scores_df = pd.read_csv(alignment_scores_filepath)
        except:
            logging.error(f'No Alignment scores for {test_name}')

        if not os.path.exists(preview_folder):
                    os.makedirs(preview_folder)

        # Here we want to create an html file for each test! The html file will contain the results plots of its own test!

        test_html_filename = test_name.lower().replace(' - ', '_').replace(' (', '_').replace(')', '').replace(' ', '_').replace('-', '_')+'.html'
        hr_test = html_report(os.path.join(output_folder, test_html_filename), header_file=header_file)

        test_preview_html_filename = test_name.lower().replace(' - ', '_').replace(' (', '_').replace(')', '').replace(' ', '_').replace('-', '_')+'_preview.html'
        hr_test_preview = html_report(os.path.join(output_folder, test_preview_html_filename), header_file=header_file)

        js_filename = 'script_'+test_name.lower().replace(' - ', '_').replace(' (', '_').replace(')', '').replace(' ', '_').replace('-', '_')+'.js'
        js_main = js_report(os.path.join(output_folder, 'js_files', js_filename))

        hr_test.insert_file(os.path.join('test_reports_utils', "nav.html"))


        hr_test.beg_div(class_tag='other_content')
        hr_test.insert_header(test_name)

        hr_test.insert_text('The plots are generated with Plotly JS, please wait for few seconds for all the plots to be generated.')
        hr_test.insert_text('The countour plots are dynamic, you can hover over them and read the results, or zoom in and zoom out.')
        
        if test_name != 'Contrast Constancy - Contrast Matching':
            hr_test.insert_text('Each contour plot corresponds to the metric "quality" predictions for the "{the_test_name}" test, while the red line corresponds to the ground truth. The test stimuli examples can be found <a href={link}>here</a>.'.format(link=test_preview_html_filename, the_test_name=test_name))
        else:
            hr_test.insert_text('Each plot corresponds to the results from the "Contrast Matching" experiment, where different colors represent different contrast C<sub>r</sub> values. The dashed lines are human results, while the solid lines are model-predicted results. The test stimuli examples can be found <a href={link}>here.'.format(link=test_preview_html_filename))

        # Create the preview webpage!

        hr_test_preview.beg_table()

        # Now, we want to go through each condition, get both test and ref figures, save them, and put them in the webpage!
        for i in np.linspace(0, test_rows-1, 5).astype(int):
            hr_test_preview.beg_table_row()
            for j in np.linspace(0, test_columns-1, 5).astype(int):
                hr_test_preview.beg_table_cell()

                if test_name == 'Contrast Matching - Across Spatial Frequencies':
                    condition_index = Test.N_ref_contrasts + i*test_columns + j
                else:
                    condition_index = i*test_columns+j

                test_filename = os.path.join(preview_folder, 'test_' + str(condition_index))
                ref_filename = os.path.join(preview_folder, 'ref_' + str(condition_index))

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

                    hr_test_preview.beg_div(class_tag='cell-content')
                    hr_test_preview.insert_image(test_filename, mousedown_image=ref_filename, class_tag="toggle-image")
                    if test_name != 'Contrast Matching - Across Cardinal Color Directions':
                        hr_test_preview.insert_text(f"{x_variable_name} = {x_variable[condition_index]:.2g}, {y_variable_name} = {y_variable[condition_index]:.2g}")
                    else:
                        hr_test_preview.insert_text(f"{x_variable_name} = {x_variable[condition_index]:.2g}, {y_variable_name} = {y_variable[condition_index]}")
                    hr_test_preview.close_div()

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

                    hr_test_preview.beg_div(class_tag='cell-content')
                    hr_test_preview.insert_video(test_filename, mousedown_video=ref_filename, class_tag="toggle-video")
                    hr_test_preview.insert_text(f"{x_variable_name} = {x_variable[condition_index]:.2g}, {y_variable_name} = {y_variable[condition_index]:.2g}")
                    hr_test_preview.close_div()
                    

                hr_test_preview.end_table_cell()
            hr_test_preview.end_table_row()
        hr_test_preview.end_table()

        # Now, we want to add all metric results to this test webpage

        hr_test.beg_div(class_tag='image-container')

        for j, M in enumerate(metrics):

            try:

                metric_name = M.name()

                metric_id = test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')+'_'+metric_name.lower().replace(' ', '_')

                metric_predictions_file = os.path.join(results_folder, metric_name+'.csv')
                metric_predictions_df = pd.read_csv(metric_predictions_file)

                predictions = metric_predictions_df['Prediction'].to_numpy()
                try:
                    alignment_score = alignment_scores_df[alignment_scores_df['Metric'] == metric_name]['Score'].to_numpy()[0]
                    title = metric_name + ' Metric<br>Alignment Score = '+str(round(np.abs(alignment_score), 3))
                except:
                    title = metric_name + ' Metric'

                if generate_metrics_results:
                    js_command = Test.plotly(predictions, reverse_color_order=M.is_lower_better(), title=title, fig_id=metric_id)

                # Similarly to the main html file we need to create a JS file!

                js_main.println(js_command)

                hr_test.beg_div(id_tag = metric_id)
                hr_test.close_div()

            except:
                logging.error(f'Failed on test {test_name} and the {metric_name} metric')

        hr_test.close_div()
        hr_test.close_div()

        hr_test.close(js_filename)
        js_main.close()
    
    hr_main.close()





