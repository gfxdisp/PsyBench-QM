import pandas as pd
import logging
from tqdm import trange
import shutil

from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.tests_utils import get_all_tests
from metrics.piq_metrics import *
from metrics.pyiqa_metrics import *

from test_reports_utils.html_report import html_report


if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    tests = get_all_tests()
    tests.append(ContrastMakingAchTest)

    input_folder = 'results'
    output_folder = 'synthetic_test_webpage'

    generate_stimuli = False
    generate_metrics_results = True


    metrics = [dss_metric, fsim_metric, gmsd_metric, haarpsi_metric, iw_ssim_metric, mdsi_metric, ms_gmsd_metric,
               ms_ssim_metric, ssim_metric, vif_metric, vsi_metric, mad_metric, psnr_metric]

    # Create the directory and its associated html file!

    #if not os.path.exists(output_folder):
    #os.makedirs(os.path.join(output_folder, 'html_files'))
    #os.makedirs(os.path.join(output_folder, 'previews'))
    #os.makedirs(os.path.join(output_folder, 'plots'))
    #os.makedirs(os.path.join(output_folder, 'css_files'))

    shutil.copyfile(os.path.join('test_reports_utils', 'style.css'), os.path.join(output_folder, 'css_files', 'style.css'))

    header_file = os.path.join('test_reports_utils', "header.html")
    hr_main = html_report(os.path.join(output_folder, 'index.html'), header_file=header_file )

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
        for i in range(test_rows):
            hr_test.beg_table_row()
            for j in range(test_columns):
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
                    hr_test.insert_text('{x_var_name} = {x_var}, {y_var_name} = {y_var}'.format(x_var_name=x_variable_name, y_var_name=y_variable_name, x_var=x_variable[condition_index], y_var=y_variable[condition_index]))
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
                    hr_test.insert_text('{x_var_name} = {x_var}, {y_var_name} = {y_var}'.format(x_var_name=x_variable_name,y_var_name=y_variable_name,x_var=x_variable[condition_index],y_var=y_variable[condition_index]))
                    hr_test.close_div()
                    

                hr_test.end_table_cell()
            hr_test.end_table_row()
        hr_test.end_table()

        hr_main.beg_div(class_tag='image-container')

        for j, M in enumerate(metrics):

            try:
                Metric = M()
                metric_name = Metric.short_name()

                metric_predictions_file = os.path.join(results_folder, metric_name+'.csv')

                metric_predictions_df = pd.read_csv(metric_predictions_file)

                predictions = metric_predictions_df['Prediction'].to_numpy()

                output_filename = os.path.join(plot_folder, metric_name+'.png')

                if generate_metrics_results:
                    Test.plot(predictions, output_filename=output_filename, title=metric_name+' Metric')

                # Now add them to the webpage

                output_filename = os.path.normpath(output_filename).split(os.path.sep)[1:]
                output_filename = os.path.join(*output_filename)

                hr_main.insert_image(output_filename)

            except:
                logging.error(f'Failed on test {test_name} and the {metric_name} metric')

        hr_main.close_div()

        hr_test.close()

    hr_main.close()





