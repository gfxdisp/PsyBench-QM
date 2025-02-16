from math import log10, log2
import abc
from pyst.synthetic_test import SyntheticTest
import numpy as np
import os
import matplotlib.pyplot as plt
from pyst.utils import *
import torch 
from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
import json


class ContrastConstancySpatialFreqAchTest(SyntheticTest):
    """
    The test of contrast constancy across spatial frequencies
    """

    # For now we may not need these informations! It is not as important

    def __init__(self, width=512, height=512, N_contrasts=20, N_freq=20, device=None):


        super().__init__(device)

        self.data_folder = 'pyst/pyst_data/contrast_constancy'

        self.width = width
        self.height = height

        self.dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)

        self.bkg_luminance = 10
        self.display_photometry = vvdp_display_photo_eotf(Y_peak=46.72, contrast=1000, source_colorspace='sRGB')
        self.display_geometry = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=60)

        self.ppd = self.display_geometry.get_ppd()
        self.frames = 1
        self.fps = 1

        df = json.load(open(os.path.join(self.data_folder, 'contrast_constancy_sin_5_cpd.json')))
        ref_contrasts = torch.tensor(df['average_reference_contrast'], device=self.device)
        self.ref_contrasts = torch.cat((torch.tensor([1.], device=self.device), ref_contrasts))

        self.test_contrasts = torch.logspace(-3, 0, N_contrasts).to(self.device)

        self.ref_spatial_freq = 5
        self.test_spatial_frequencies = torch.logspace(-2, log2(25), N_freq, base=2).to(self.device)
        self.ref_spatial_frequencies = torch.tensor([0.25, 0.5, 1, 2, 5, 10, 15, 20, 25], device=self.device)

    
        self.N_test_contrasts = len(self.test_contrasts)
        self.N_test_s_freq = len(self.test_spatial_frequencies)

        self.N_ref_contrasts = len(self.ref_contrasts)
        self.N_ref_s_freq = len(self.ref_spatial_frequencies)

        self.test_contrast_indices, self.test_s_freq_indices = torch.meshgrid(torch.arange(self.N_test_contrasts), torch.arange(self.N_test_s_freq), indexing='ij')

        self.test_contrast_indices = self.test_contrast_indices.flatten()
        self.test_s_freq_indices = self.test_s_freq_indices.flatten()

        self._preview_folder = 'contrast_constancy_spatial_freq_ach_tests'
        self._preview_as = 'image'


    def __len__(self):
        return self.N_test_contrasts*self.N_test_s_freq + self.N_ref_contrasts

    def get_test_condition_parameters(self, index):

        if index < self.N_ref_contrasts:
            contrast = self.ref_contrasts[index]
            s_freq = self.ref_spatial_freq
        
        else:
            new_index = index - self.N_ref_contrasts
            contrast = self.test_contrasts[self.test_contrast_indices[new_index]]
            s_freq = self.test_spatial_frequencies[self.test_s_freq_indices[new_index]]

        return contrast, s_freq
    
    def get_test_condition(self, index):

        contrast, s_freq= self.get_test_condition_parameters(index)
        L = self.bkg_luminance
        Lmax = self.display_photometry.get_peak_luminance()

        XYZ_test, XYZ_reference = generate_sinusoidal_grating(self.width, self.height, self.frames, self.ppd, self.fps, contrast, s_freq, L, self.dkl_col_direction, device=self.device)

        test_condition = lin2srgb(XYZ2RGB709_nd(XYZ_test / Lmax).unsqueeze(0))
        reference_condition = lin2srgb(XYZ2RGB709_nd(XYZ_reference / Lmax).unsqueeze(0))

        return video_source.video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW", display_photometry=self.display_photometry)


    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as
    
    def get_ticks(self):
        x = torch.logspace(-2, 4, 7, base=2)
        y = [0.001, 0.01, 0.1, 1]

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Spatial Frequency']

    def get_rows_conditions(self):
        contrasts = torch.cat((self.ref_contrasts, self.test_contrasts[self.test_contrast_indices]))
        frequencies = torch.cat((torch.ones_like(self.ref_contrasts)*self.ref_spatial_freq, self.test_spatial_frequencies[self.test_s_freq_indices]))
        return [contrasts, frequencies]

    def size(self):
        return self.N_test_contrasts, self.N_test_s_freq 

    def units(self):
        return ['log', 'log'], [None, 'cpd']
    
    def get_gt_results(self):
        df = json.load(open(os.path.join(self.data_folder, 'contrast_constancy_sin_5_cpd.json')))
        ref_contrasts = df['average_reference_contrast']
        spatial_frequencies = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
        gt_contrasts = []
        for i in range(len(ref_contrasts)):
            gt_contrasts.append(df['ref_contrast_index_'+str(i)]['y_test_contrast_average'])

        return ref_contrasts, spatial_frequencies, gt_contrasts
    

    def plot(self, predictions, reverse_color_order=False, output_filename=None, title=None):

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()

        y_condition = conditions[0].reshape(self.size()).cpu().numpy()
        y_condition_name = condition_names[0]

        x_condition = conditions[1].reshape(self.size()).cpu().numpy()
        x_condition_name = condition_names[1]

        predictions = predictions.reshape(self.size())

        cmap = 'viridis'

        if reverse_color_order:
            cmap = 'viridis_r'

        fig = plt.figure(figsize=(5, 4))
        contour = plt.contour(x_condition, y_condition, predictions, 30, cmap=cmap)

        #plt.clabel(contour, inline=True, fontsize=8)

        colorbar = fig.colorbar(contour)
        if reverse_color_order:
            colorbar.ax.invert_yaxis()

        plt.grid(alpha=0.4, axis='x')

        scales, units = self.units()

        plt.xscale(scales[1])
        plt.yscale(scales[0])

        if units[1]:
            plt.xlabel(x_condition_name+' ['+units[1]+']', fontsize=12)
        else:
            plt.xlabel(x_condition_name, fontsize=12)

        if units[0]:
            plt.ylabel(y_condition_name+' ['+units[1]+']', fontsize=12)
        else:
            plt.ylabel(y_condition_name, fontsize=12)
        
        x_ticks, y_ticks = self.get_ticks()
        x_ticks = x_ticks.cpu().numpy()
        y_ticks = y_ticks.cpu().numpy()

        formatted_xticks = [f'{tick:g}' for tick in x_ticks]
        formatted_yticks = [f'{tick:.2g}' for tick in y_ticks]

        plt.xticks(x_ticks, formatted_xticks, fontsize=10)
        plt.yticks(y_ticks, formatted_yticks, fontsize=10)

        # Plot the csf results

        x_csf, y_csf = self.get_csf_results()

        #plt.plot(x_csf, y_csf+0.01, marker='o', color='red', linewidth=1, markerfacecolor='none', markeredgecolor='red')
        plt.plot(x_csf, y_csf, color='red', linewidth=1)

        plt.xlim([np.min(x_condition), np.max(x_condition)])
        plt.ylim([np.min(y_condition), np.max(y_condition)])

        if title is not None:
            plt.title(title, fontsize=14)

        plt.tight_layout()

        if output_filename is not None:
            fig.savefig(output_filename)

        plt.close(fig)
    
    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""        

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()
        units_scales, units_names = self.units()

        # Do the matching!

        ref_contrasts = conditions[0][1:self.N_ref_contrasts].cpu().numpy()
        ref_contrasts_predictions = predictions[1:self.N_ref_contrasts] / predictions[0]

        test_contrasts = conditions[0][self.N_ref_contrasts:].cpu().numpy()
        test_frequencies = conditions[1][self.N_ref_contrasts:].cpu().numpy()
        predictions = predictions[self.N_ref_contrasts:]

        matched_test_contrasts = []

        for i, ref_contrast in enumerate(ref_contrasts):
            ref_prediction = ref_contrasts_predictions[i]
            indices = []
            for test_freq in np.unique(test_frequencies):
                test_freq_contrasts = test_contrasts[test_frequencies == test_freq]

                test_freq_predictions = predictions[test_frequencies == test_freq]
                test_freq_predictions = test_freq_predictions / test_freq_predictions[-1]

                index = np.argmin((ref_prediction - test_freq_predictions)**2)
                indices.append(test_freq_contrasts[index])
            
            matched_test_contrasts.append(indices)

        matched_test_contrasts = np.array(matched_test_contrasts)
        test_frequencies = np.unique(test_frequencies)


        # Get Ground Truth

        _, _, matched_ref_contrasts = self.get_gt_results()
        matched_ref_contrasts = np.array(matched_ref_contrasts)

        # Now Making the figure!
        

        y_condition = np.unique(conditions[0].cpu().numpy())
        y_condition_name = condition_names[0]
        if units_names[0] is not None:
            y_condition_name += " ["+units_names[0]+"]"


        x_condition = np.unique(conditions[1].cpu().numpy())
        x_condition_name = condition_names[1]
        if units_names[1] is not None:
            x_condition_name += " ["+units_names[1]+"]"

        z_condition_name = "Metric Prediction"

        predictions = predictions.reshape(self.size())

        x_tick_vals, y_tick_vals = self.get_ticks()
        x_tick_vals = x_tick_vals.cpu().numpy()

        z = "["

        for i, row in enumerate(predictions):
            if i>0:
                z += ", "
            z += "["
            for j, el in enumerate(row):
                if j > 0:
                    z += ", "
                z += str(el)
            z += "]"

        z += "]"

        x = "["
        for i, el in enumerate(x_condition):
            if i>0:
                x += ", "
            x += str(el)
        x += "]"

        y = "["
        for i, el in enumerate(y_condition):
            if i>0:
                y += ", "
            y += str(el)
        y += "]"

        # Contour plot data

        js_command += "var Contourdata = {\n"
        js_command += "z: " + z + ",\nx: " + x + ",\ny: " + y

        extra_params = "type: 'contour',\ncolorscale: 'Viridis',\nncontours: 30,\nline:{width:2},\n"
        if reverse_color_order:
            extra_params+= "reversescale: true,\n"

        extra_params += "hovertemplate: '"+x_condition_name+": %"+r"{x:.2f}"+"<br>"+y_condition_name+": %"+r"{y:.2f}"+"<br>"+z_condition_name+": %"+r"{z:.2f}"+"<extra></extra>',\n"
        
        extra_params += r"contours:{coloring:'lines'}"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        # Ground Truth Data 

        x_gt, y_gt = self.get_csf_results()

        x = "["
        for i, el in enumerate(x_gt):
            if i>0:
                x += ", "
            x += str(el.cpu().numpy())
        x += "]"

        y = "["
        for i, el in enumerate(y_gt):
            if i>0:
                y += ", "
            y += str(el)
        y += "]"

        js_command += "\nvar Linedata = {\n"
        js_command += "x: " + x + ",\ny: " + y

        extra_params = "mode: 'lines',\ntype: 'scatter',\nline:{color: 'red', width:2},marker:{symbol:'circle-open', size:8},\nname: 'CastleCSF'"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        js_command += "\nvar data = [Contourdata, Linedata]"

        # Layout of the Figure

        x = "["
        for i, el in enumerate(x_tick_vals):
            if i>0:
                x += ", "
            x += str(el)
        x += "]"

        y = "["
        for i, el in enumerate(y_tick_vals):
            if i>0:
                y += ", "
            y += str(el)
        y += "]"

        axes_info = "xaxis: {type:'log', range: ["+str(log10(np.min(x_condition)))+","+str(log10(np.max(x_condition)))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, yaxis: {type:'log', range:["+str(log10(np.min(y_condition)))+","+str(log10(np.max(y_condition)))+"], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', data, layout);\n\n"

        return js_command

    def short_name(self):
        return 'Contrast Matching - Across Spatial Frequencies'


