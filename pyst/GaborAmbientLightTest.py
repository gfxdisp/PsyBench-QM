from math import log10
import os
import numpy as np
import torch

from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
from pycvvdp import video_source

# Reuse helpers from pyst.utils
from pyst.utils import *

# Import the abstract base class to inherit common behavior
from pyst.GaborPatch import ContrastDetectionTest


class GaborAmbientLightTest(ContrastDetectionTest):
    """Gabor patch test that varies ambient illuminance (lux) and contrast.

    Background luminance (the patch/background level used to generate the
    stimuli) is fixed (default 1 nit). Ambient illuminance (lux) is treated
    separately and passed into the display model per-condition via the
    E_ambient parameter.

    Args:
        width, height: stimulus size in pixels
        N_contrasts: number of contrast steps (logspace 0.01->1)
        N_lux: number of ambient illuminance steps (logspace 0.1->1e5)
        bg_luminance: background luminance in cd/m^2 (nits). Default 1.
    """

    def __init__(self, width=1920, height=1080, N_contrasts=20, N_lux=20, bg_luminance=1.0, is_alignmenet_score=False, device=None):
        channel = 'Ach'
        frames = 1
        fps = 1
        stimuli_type = 'gabor_patch'

        # Contrast: 0.01 -> 1 (10^-2 .. 10^0)
        contrasts = torch.logspace(-2, 0, N_contrasts)

        # Ambient illuminance values (lux): 0.1 -> 100000 (10^-1 .. 10^5)
        ambient_lux = torch.logspace(-1, 5, N_lux)

        # Base HDR display settings (peak luminance). We'll create per-condition
        # display objects later with different E_ambient values.
        base_dm = vvdp_display_photo_eotf(Y_peak=10000, source_colorspace='BT.709', EOTF="linear", E_ambient=0, contrast=10000000)
        gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=66)

        # Call the base constructor with fixed background luminance (single value)
        super().__init__(width, height, channel, base_dm, gm, frames, fps,
                         contrasts, bkg_luminances=10, sizes=2,
                         stimuli_type=stimuli_type, is_alignmenet_score=is_alignmenet_score, device=device)

        # Store ambient values separately and the chosen fixed background luminance
        self.ambient_lux = ambient_lux.to(self.device)
        self.N_ambient = len(self.ambient_lux)

        # Adjust preview folder and format
        self._preview_folder = os.path.join(self._preview_folder, 'ambient_light')
        self._preview_as = 'image'

    # Override rows/size helpers to expose (contrast x ambient) grid
    def get_row_header(self):
        return ['Contrast', 'Ambient Illumination']

    def get_rows_conditions(self):
        # Return two tensors flattened in the same ordering used by plot routines
        C, A = torch.meshgrid(self.contrasts.cpu(), self.ambient_lux.cpu(), indexing='ij')
        return [C.flatten(), A.flatten()]

    def size(self):
        return len(self.contrasts), self.N_ambient
    
    def __len__(self):
        return len(self.contrasts) * self.N_ambient

    def units(self):
        return ['log', 'log'], [None, 'lux']

    def get_ticks(self):
        # contrast ticks and ambient ticks for plotting guidance
        x = [0.01, 0.1, 1]
        y = [0.1, 1, 10, 100, 1000, 10000, 100000]
        return torch.tensor(y), x

    def short_name(self):
        return 'Gabor - Ambient Illuminance vs Contrast'

    def latex_name(self):
        return 'Gabor - Ambient Illuminance vs Contrast'

    def get_test_condition_parameters(self, index):
        # Map linear index to (contrast_idx, ambient_idx)
        c_idx = index // self.N_ambient
        a_idx = index % self.N_ambient

        contrast = self.contrasts[c_idx]
        s_freq = self.spatial_frequencies[self.s_freq_indices[0]] if self.N_s_freq > 0 else self.spatial_frequencies[0]
        t_freq = self.temporal_frequencies[self.t_freq_indices[0]] if self.N_t_freq > 0 else self.temporal_frequencies[0]
        L = self.bkg_luminances[self.L_indices[0]] if self.N_L > 0 else self.bkg_luminances[0]
        orien = self.orientations[self.orien_indices[0]] if self.N_orien > 0 else self.orientations[0]
        size = self.sizes[self.size_indices[0]] if self.N_size > 0 else self.sizes[0]
        ecc = self.eccentricities[self.ecc_indices[0]] if self.N_ecc > 0 else self.eccentricities[0]
        ambient = self.ambient_lux[a_idx]

        return contrast, s_freq, t_freq, L, orien, size, ecc, ambient

    def get_test_condition(self, index):
        # Build stimulus using fixed background luminance but per-condition ambient
        contrast, s_freq, t_freq, L, orien, size, ecc, ambient = self.get_test_condition_parameters(index)

        # Create a local display photometry with E_ambient set to this ambient lux
        dm_local = vvdp_display_photo_eotf(Y_peak=self.display_photometry.get_peak_luminance(),
                                           source_colorspace='BT.709',
                                           EOTF=getattr(self.display_photometry, 'EOTF', None) or 'linear',
                                           E_ambient=float(ambient.cpu().numpy()),
                                           contrast=getattr(self.display_photometry, 'contrast', None) or 10000000)

        Lmax = dm_local.get_peak_luminance()

        # Generate XYZ stimuli (uses luminance = background luminance L)
        if self.stimuli_type == 'gabor_patch':
            XYZ_test, XYZ_reference = generate_gabor_patch(self.width, self.height, self.frames, self.ppd, self.fps, contrast, s_freq, t_freq, orien, L, self.dkl_col_direction, size, ecc, device=self.device)
        elif self.stimuli_type == 'band_limited_noise':
            XYZ_test, XYZ_reference = generate_band_limited_noise(self.width, self.height, self.ppd, contrast, s_freq, self.band_width, L, self.dkl_col_direction, self.frames, device=self.device)
        elif self.stimuli_type == 'disk':
            XYZ_test, XYZ_reference = generate_disk(self.width, self.height, self.frames, self.ppd, self.fps, contrast, s_freq, t_freq, orien, L, self.dkl_col_direction, size, ecc, device=self.device)

        # Convert to RGB depending on display EOTF as in base class
        if dm_local.EOTF == 'sRGB':
            test_condition = lin2srgb(XYZ2RGB709_nd(XYZ_test / Lmax).unsqueeze(0))
            reference_condition = lin2srgb(XYZ2RGB709_nd(XYZ_reference / Lmax).unsqueeze(0))

        elif dm_local.EOTF in ['linear']:
            test_condition = XYZ2RGB709_nd(XYZ_test).unsqueeze(0)
            reference_condition = XYZ2RGB709_nd(XYZ_reference).unsqueeze(0)

        return video_source.video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW", display_photometry=dm_local)

    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()
        units_scales, units_names = self.units()


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
                if not np.isnan(el):
                    z += str(el)
                else:
                    z+= 'null'
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

        extra_params += "hovertemplate: '"+x_condition_name+": %"+r"{x:.3f}"+"<br>"+y_condition_name+": %"+r"{y:.4f}"+"<br>"+z_condition_name+": %"+r"{z:.4f}"+"<extra></extra>',\n"
        
        extra_params += r"contours:{coloring:'lines'}"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        # # Ground Truth Data 

        # x_gt, y_gt = self.get_csf_results()

        # x = "["
        # for i, el in enumerate(x_gt):
        #     if i>0:
        #         x += ", "
        #     x += str(el.cpu().numpy())
        # x += "]"

        # y = "["
        # for i, el in enumerate(y_gt):
        #     if i>0:
        #         y += ", "
        #     y += str(el)
        # y += "]"

        # js_command += "\nvar Linedata = {\n"
        # js_command += "x: " + x + ",\ny: " + y

        # extra_params = "mode: 'lines',\ntype: 'scatter',\nline:{color: 'red', width:2},marker:{symbol:'circle-open', size:8},\nname: 'CastleCSF'"

        # js_command += ",\n" + extra_params

        # js_command += "\n};"

        # js_command += "\nvar data = [Contourdata, Linedata]"

        js_command += "\nvar data = [Contourdata]"

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

        if units_scales[1] == 'log':
            axes_info = "xaxis: {type:'log', range: ["+str(log10(np.min(x_condition)))+","+str(log10(np.max(x_condition)))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, "
        else:
            axes_info = "xaxis: {range: ["+str(np.min(x_condition))+","+str(np.max(x_condition))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, "
        
        if units_scales[0] == 'log':
            axes_info += "yaxis: {type:'log', range:["+str(log10(np.min(y_condition)))+","+str(log10(np.max(y_condition)))+"], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"
        else:
            axes_info += "yaxis: {range:["+str(np.min(y_condition))+","+str(np.max(y_condition))+"], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"

        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', data, layout);\n\n"

        return js_command


