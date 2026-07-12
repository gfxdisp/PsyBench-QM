
from math import log10
import abc
from pyst.synthetic_test import SyntheticTest
import numpy as np
import os
import matplotlib.pyplot as plt
from pyst.utils import *
import torch 
from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
from metrics.glare_model import GlareModel

class GlareTest(SyntheticTest):

    def __init__(self, apply_glare_model=False, mtf_model='hdrvdp', width=1920, height=1080, N_patches=20, N_L=20, device=None):

        super().__init__(device)

        self.width = width
        self.height = height

        self.dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)
        
        self.display_photometry = vvdp_display_photo_eotf(Y_peak=10000, source_colorspace='BT.709', EOTF="linear", E_ambient=0, contrast=10000000)
        self.display_geometry = vvdp_display_geometry([1920, 1080], distance_m=0.6, diagonal_size_inches=15)

        self.ppd = self.display_geometry.get_ppd()

        self.contrast = 0.05
        self.s_freq = 4 # cycles per degree

        self.patches = torch.arange(N_patches, device=self.device)
        self.bck_luminances = torch.logspace(-1, log10(1000), steps=N_L, device=self.device)  # From 0.1 to 1000 cd/m2

        self.N_patches = N_patches
        self.N_L = N_L

        self.patch_indices, self.L_indices = torch.meshgrid(torch.arange(self.N_patches), torch.arange(self.N_L), indexing='ij')

        self.patch_indices = self.patch_indices.flatten()
        self.L_indices = self.L_indices.flatten()

        self._preview_folder = 'glare'
        self._preview_as = 'image'

        self.apply_glare_model = apply_glare_model
        self.mtf_model = mtf_model


    def __len__(self):
        return self.N_patches * self.N_L

    def get_test_condition_parameters(self, index):

        contrast = self.contrast
        patch = self.patches[self.patch_indices[index]]
        L = self.bck_luminances[self.L_indices[index]]

        return contrast, patch, L

    def get_test_condition(self, index):

        contrast, patch, L = self.get_test_condition_parameters(index)
        Lmax = self.display_photometry.get_peak_luminance()

        # First, generate the reference stimulus (the background only)

        wp_d65 = torch.tensor([0.95047, 1.0, 1.08883], device=self.device)

        DKL_gray = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * L)).view(1, 1, 3, 1)
        DKL_reference = torch.ones(self.height, self.width, 3, 1).to(self.device) * DKL_gray
        DKL_reference[:, :self.width//2, :, :] = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * Lmax)).view(1, 1, 3, 1)  # Add the glare source on the left half

        # Create our test stimulus 
        patch_width = self.width // 2 // self.N_patches

        width_deg = patch_width / self.ppd
        height_deg = self.height / self.ppd

        xx, yy = torch.meshgrid(torch.linspace(-width_deg/2, width_deg/2, patch_width).to(self.device) , torch.linspace(-height_deg/2, height_deg/2, self.height).to(self.device), indexing='xy')
        rr = xx

        grating = torch.sin(2*pi*rr*self.s_freq) * contrast * L

        ge_sigma = 0.5 * width_deg
        R2 = xx**2
        ge =  torch.exp(-R2/(2*ge_sigma**2))  # Create the gaussian envelope

        stimulus = torch.zeros_like(DKL_reference)
        stimulus[:, self.width//2+patch_width*patch:self.width//2+patch_width*(patch+1), :, :] = (grating * ge).view(self.height, patch_width, 1, 1)

        DKL_test = DKL_reference + stimulus * self.dkl_col_direction.view(1, 1, 3, 1)

        XYZ_reference = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_reference.permute(2, 3, 0, 1))) # FROM HWCF to CFHW
        XYZ_test = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_test.permute(2, 3, 0, 1)))

        test_condition = XYZ2RGB709_nd(XYZ_test).unsqueeze(0)
        reference_condition = XYZ2RGB709_nd(XYZ_reference).unsqueeze(0)

        vs = video_source.video_source_array(test_condition, reference_condition, fps=1, dim_order="BCFHW", display_photometry=self.display_photometry)
        if self.apply_glare_model:
            vs = GlareModel(vs, self.display_geometry, surround=None, mtf_model=self.mtf_model)

        return vs
    
    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()
        units_scales, units_names = self.units()

        y_condition = np.unique(conditions[1].cpu().numpy())
        y_condition_name = condition_names[1]
        if units_names[1] is not None:
            y_condition_name += " ["+units_names[1]+"]"

        x_condition = np.unique(conditions[0].cpu().numpy()) * self.width // 2 // self.N_patches / self.ppd
        x_condition_name = condition_names[0]
        if units_names[0] is not None:
            x_condition_name += " ["+units_names[0]+"]"

        z_condition_name = "Metric Prediction"

        predictions = predictions.reshape(self.size(), order='F')

        x_tick_vals, y_tick_vals = self.get_ticks()

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
                    z += 'null'
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

        js_command += "var data = {\n"
        js_command += "z: " + z + ",\nx: " + x + ",\ny: " + y

        extra_params = "type: 'contour',\ncolorscale: 'Viridis',\nncontours: 20,\n"
        if reverse_color_order:
            extra_params+= "reversescale: true,\n"

        extra_params += "hovertemplate: '"+x_condition_name+": %"+r"{x:.4f}"+"<br>"+y_condition_name+": %"+r"{y:.4f}"+"<br>"+z_condition_name+": %"+r"{z:.4f}"+"<extra></extra>',\n"
        
        extra_params += r"contours:{coloring:'heatmap'}"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        # Layout of the Figure

        x = "["
        for i, el in enumerate(x_tick_vals):
            if i>0:
                x += ", "
            x += str(np.round(el, 2))
        x += "]"

        y = "["
        for i, el in enumerate(y_tick_vals):
            if i>0:
                y += ", "
            y += str(el)
        y += "]"

        axes_info = "xaxis: {range: ["+str(np.min(x_condition))+","+str(np.max(x_condition))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, yaxis: {type:'log', range:["+str(log10(np.min(y_condition)))+","+str(log10(np.max(y_condition)))+"], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', [data], layout);\n\n"

        return js_command



    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as
    
    def get_ticks(self):
        x = np.linspace(0, self.N_patches-1, 6) * (self.width // 2 // self.N_patches) / self.ppd
        y = np.logspace(-1, 3, 5)

        return x, y

    def get_row_header(self):
        return ['Distance', 'Luminance']

    def get_rows_conditions(self):
        return [self.patches[self.patch_indices], self.bck_luminances[self.L_indices]]

    def size(self):
        return self.N_patches, self.N_L

    def units(self):
        return ['linear', 'log'], ['Visual degrees', 'cd/m²']

    def short_name(self):
        return 'Glare'
    
    # def latex_name(self):
    #     return 'Contrast - Detection - Luminance'

    def short_name(self):
        return 'Glare Test'



def imgaussfilt_simple(noise, sigma=3):
    # make 1D Gaussian kernel
    ksize = int(6*sigma + 1)
    if ksize % 2 == 0: ksize += 1
    coords = torch.arange(ksize, device=noise.device) - ksize // 2
    g = torch.exp(-(coords**2)/(2*sigma**2))
    g /= g.sum()
    
    # 2D kernel
    kernel = g[:, None] * g[None, :]
    kernel = kernel[None, None, :, :]  # shape (1,1,H,W)
    
    # apply convolution
    return torch.nn.functional.conv2d(noise[None, None, :, :], kernel, padding=ksize//2).squeeze()


def make_bandpass_noise(height, width, noise_freq_cpd, ppd, device, seed=12345):
    torch.manual_seed(seed)

    noise = torch.randn(height, width, device=device)
    noise_f = torch.fft.fft2(noise)

    f0 = float(noise_freq_cpd) / float(ppd)
    freq_edge_low = 2 ** (np.log2(f0) - 0.5)
    freq_edge_high = 2 ** (np.log2(f0) + 0.5)

    fx = torch.fft.fftfreq(width, d=1.0, device=device)[None, :]
    fy = torch.fft.fftfreq(height, d=1.0, device=device)[:, None]
    radius = torch.sqrt(fx * fx + fy * fy)
    mask = (radius >= freq_edge_low) & (radius <= freq_edge_high)

    noise = torch.fft.ifft2(noise_f * mask).real
    noise /= noise.std()

    return noise


class GlareDiscTest(SyntheticTest):

    def __init__(self, apply_glare_model=False, mtf_model='hdrvdp', width=2048, height=1536, N_contrasts=20, N_multiplier=10, is_alignmenet_score=False , device=None):


        super().__init__(device)

        self.width = width
        self.height = height
        self.frames = 1

        self.dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)

        self.display_photometry = vvdp_display_photo_eotf(Y_peak=5000, source_colorspace='BT.709', EOTF="linear", E_ambient=0, contrast=1000000)
        self.display_geometry = vvdp_display_geometry([2048, 1536], distance_m=0.94, diagonal_size_inches=9.7)

        self.ppd = self.display_geometry.get_ppd()

        self.is_alignmenet_score = is_alignmenet_score

        if self.is_alignmenet_score:
            self.contrasts = torch.logspace(log10(0.5), log10(2), N_multiplier).to(self.device)
        else:
            self.contrasts = torch.logspace(-2, 0, N_contrasts, device=self.device)

        self.radical_distances = torch.tensor([0.2, 0.9, 1.6], device=self.device)  # in degrees
    
        self.N_contrasts = len(self.contrasts)
        self.N_distances = len(self.radical_distances)

        self.contrast_indices, self.distance_indices = torch.meshgrid(torch.arange(self.N_contrasts), torch.arange(self.N_distances), indexing='ij')

        self.contrast_indices = self.contrast_indices.flatten()
        self.distance_indices = self.distance_indices.flatten()


        self._preview_folder = os.path.join('glare_tests', 'glare_disc_tests')

        self._preview_as = 'image'

        self.csf_data_folder = 'pyst/pyst_data/disc_glare_results/'

        if self.is_alignmenet_score:
            self.x_variable = self.radical_distances
            self.x_variable_indices = self.distance_indices
            
            self.x_gt, self.y_gt = self.get_csf_results()
            self.x_gt = self.x_gt.cpu().numpy()
        
        self.apply_glare_model = apply_glare_model
        self.mtf_model = mtf_model


    def __len__(self):
        return self.N_contrasts*self.N_distances

    def get_csf_results(self):
        x = self.radical_distances
        # y = np.loadtxt(os.path.join(self.csf_data_folder, 'pilot_31_12_25.txt'))
        y = 10 ** np.array([-0.426952199606121, -0.798045004976647, -0.955546315668485])
        return x, y


    def get_test_condition_parameters(self, index):

        distance = self.radical_distances[self.distance_indices[index]]

        if self.is_alignmenet_score:
            multiplier = self.contrasts[self.contrast_indices[index]]
            x_var = self.x_variable[self.x_variable_indices[index]]
            """ Using both the multiplier and the x_var_index, we can identify the contrast to be used!"""
            sensitivity_gt = 1 / np.interp(x_var.cpu(), self.x_gt, self.y_gt)
            contrast = 1 / (sensitivity_gt * multiplier)
        else:
            contrast = self.contrasts[self.contrast_indices[index]]
        

        return contrast, distance
    
    def get_test_condition(self, index):

        contrast, distance = self.get_test_condition_parameters(index)
        Lmax = self.display_photometry.get_peak_luminance()
        background_L = 1.0  # cd/m2
        RGB_gray = XYZ2RGB709_nd(torch.tensor([0.95047, 1.0, 1.08883], device=self.device).view(1, 1, 3, 1) )

        # We get our stimuli in Rec709 RGB format
        disk_radius = 1.0 * self.ppd  # 1 degree in pixels

        # Get the reference stimulus 
        reference_condition = torch.ones(self.height, self.width, device=self.device) * background_L

        xx, yy = torch.meshgrid(torch.linspace(-self.width/2, self.width/2, self.width).to(self.device) , torch.linspace(-self.height/2, self.height/2, self.height).to(self.device), indexing='xy')
        rr = torch.sqrt(xx**2 + yy**2)

        disc = (rr <= disk_radius).float() * Lmax 

        reference_condition = (reference_condition + disc).clamp(0, Lmax)

        # Now, create the test stimulus
        if distance == 0.2:
            noise_inner_radius = 1.0 * self.ppd
            noise_outer_radius = 1.4 * self.ppd
        elif distance == 0.9:
            noise_inner_radius = 1.7 * self.ppd
            noise_outer_radius = 2.1 * self.ppd
        elif distance == 1.6:
            noise_inner_radius = 2.4 * self.ppd
            noise_outer_radius = 2.8 * self.ppd
        
        orientation = torch.tensor(0.0)  # Radians

        # Set noise seed for reproducibility
        torch.manual_seed(12345)

        # Create the noise pattern
        noise = torch.randn(self.height, self.width, device=self.device) # Generate white noise with mean 0 and std 1
        noise = imgaussfilt_simple(noise, sigma=3)  # Low-pass filter the noise to make it smoother
        noise /= noise.std()  # Normalize to have std 1

        # Get mask for annulus
        perp_proj = -xx * torch.sin(orientation) + yy * torch.cos(orientation)
        stripe_mask = (torch.abs(perp_proj) <= (disk_radius))

        mask = ((rr > noise_inner_radius) & (rr <= noise_outer_radius) & stripe_mask).float()

        noise = noise * mask * contrast

        # Create the test stimulus
        test_condition = (reference_condition + noise).clamp(0, Lmax)

        # Apply color space transformation
        reference_condition = reference_condition.view(self.height, self.width, 1, 1) * RGB_gray
        test_condition = test_condition.view(self.height, self.width, 1, 1) * RGB_gray

        vs = video_source.video_source_array(test_condition, reference_condition, 1, dim_order="HWCF", display_photometry=self.display_photometry)
        
        if self.apply_glare_model:
            vs = GlareModel(vs, self.display_geometry, surround=None, mtf_model=self.mtf_model)
        
        return vs


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

    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry
    
    def get_ticks(self):
        x = self.radical_distances
        y = [0.001, 0.01, 0.1, 1]

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Distance from Glare Source']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.radical_distances[self.distance_indices]]

    def size(self):
        return self.N_contrasts, self.N_distances

    def units(self):
        return ['log', 'linear'], [None, 'visual degrees']

    def short_name(self):
        return 'Contrast Detection - Achromatic Gaussian noise - Disc glare source'
    
    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as

    def short_name(self):
        return 'Noise detection with Glare Disc'


class GlareRingTest(GlareDiscTest):

    def __init__(self, apply_glare_model=False, mtf_model='hdrvdp', width=2048, height=1536, N_contrasts=20, N_multiplier=10, is_alignmenet_score=False, device=None):
        super().__init__(apply_glare_model=apply_glare_model, mtf_model=mtf_model, width=width, height=height, N_contrasts=N_contrasts, N_multiplier=N_multiplier, is_alignmenet_score=is_alignmenet_score, device=device)
        self.background_lum = 2.0
        self.noise_radius = 0.7
        self.glare_size = 1.0
        self.noise_freq = 4
        self.radical_distances = torch.tensor([0.3, 1.3, 2.3], device=self.device)
        self.N_distances = len(self.radical_distances)
        self.distance_indices = torch.arange(self.N_distances, device=self.device).repeat(self.N_contrasts)
        self.contrast_indices = torch.arange(self.N_contrasts, device=self.device).repeat_interleave(self.N_distances)
        self._preview_folder = os.path.join('glare_tests', 'glare_ring_tests')

    def get_test_condition(self, index):

        contrast, distance = self.get_test_condition_parameters(index)
        Lmax = self.display_photometry.get_peak_luminance()
        background_L = self.background_lum  # cd/m2
        RGB_gray = XYZ2RGB709_nd(torch.tensor([0.95047, 1.0, 1.08883], device=self.device).view(1, 1, 3, 1))

        # The MATLAB generator uses a bright ring around a central noise disk.
        noise_radius = self.noise_radius * self.ppd
        glare_inner_radius = noise_radius + distance * self.ppd
        glare_outer_radius = glare_inner_radius + self.glare_size * self.ppd
        glare_edge_width = max(1, round(0.15 * self.ppd))

        xx, yy = torch.meshgrid(
            torch.linspace(-self.width / 2, self.width / 2, self.width).to(self.device),
            torch.linspace(-self.height / 2, self.height / 2, self.height).to(self.device),
            indexing='xy'
        )
        rr = torch.sqrt(xx**2 + yy**2)

        reference_condition = torch.ones(self.height, self.width, device=self.device) * background_L

        glare = torch.zeros_like(reference_condition)
        core_mask = (rr > glare_inner_radius + glare_edge_width) & (rr <= glare_outer_radius - glare_edge_width)
        glare[core_mask] = Lmax

        inner_falloff_mask = (rr > glare_inner_radius) & (rr <= glare_inner_radius + glare_edge_width)
        glare[inner_falloff_mask] = Lmax * (rr[inner_falloff_mask] - glare_inner_radius) / glare_edge_width

        outer_falloff_mask = (rr > glare_outer_radius - glare_edge_width) & (rr <= glare_outer_radius)
        glare[outer_falloff_mask] = Lmax * (glare_outer_radius - rr[outer_falloff_mask]) / glare_edge_width

        reference_condition = (reference_condition + glare).clamp(0, Lmax)

        noise = make_bandpass_noise(self.height, self.width, self.noise_freq, self.ppd, self.device, seed=12345)
        noise_mask = (rr <= noise_radius)
        noise = noise * noise_mask * contrast * background_L

        test_condition = (reference_condition + noise).clamp(0, Lmax)

        reference_condition = reference_condition.view(self.height, self.width, 1, 1) * RGB_gray
        test_condition = test_condition.view(self.height, self.width, 1, 1) * RGB_gray

        vs = video_source.video_source_array(test_condition, reference_condition, 1, dim_order="HWCF", display_photometry=self.display_photometry)

        if self.apply_glare_model:
            vs = GlareModel(vs, self.display_geometry, surround=None, mtf_model=self.mtf_model)

        return vs

    def get_csf_results(self):
        x = self.radical_distances
        y = 10 ** np.array([-1.13900410986717, -1.27958488666643, -1.45563766255476])
        # y = 10 ** np.array([-1.07502019552948, -1.25661508266155, -1.48540280866984])
        return x, y

    def get_preview_folder(self):
        return self._preview_folder

    def short_name(self):
        return 'Noise detection with Glare Ring'



class GlareRingLuminanceTest(GlareRingTest):

    def __init__(self, apply_glare_model=False, mtf_model='hdrvdp', width=2048, height=1536, N_contrasts=20, N_multiplier=10, is_alignmenet_score=False, device=None):
        super().__init__(apply_glare_model=apply_glare_model, mtf_model=mtf_model, width=width, height=height, N_contrasts=N_contrasts, N_multiplier=N_multiplier, is_alignmenet_score=is_alignmenet_score, device=device)

        if self.is_alignmenet_score:
            self.contrasts = torch.logspace(log10(0.5), log10(2), N_multiplier).to(self.device)
        else:
            self.contrasts = torch.logspace(-2, log10(2), N_contrasts, device=self.device)

        # Override display geometry with correct viewing distance (1.32 m)
        self.display_geometry = vvdp_display_geometry([2048, 1536], distance_m=1.32, diagonal_size_inches=9.7)
        self.ppd = self.display_geometry.get_ppd()

        # Fixed geometry: inner and outer radius (degrees)
        self.glare_inner_radius_deg = 0.2
        self.glare_outer_radius_deg = 1.0

        # Luminance levels to sweep (cd/m2)
        self.ring_luminances = torch.tensor([0.5, 5.0, 50.0, 500.0, 5000.0], device=self.device)

        self.N_levels = len(self.ring_luminances)

        # Build index grids for contrasts x luminance levels
        self.contrast_indices, self.level_indices = torch.meshgrid(torch.arange(self.N_contrasts, device=self.device), torch.arange(self.N_levels, device=self.device), indexing='ij')
        self.contrast_indices = self.contrast_indices.flatten()
        self.level_indices = self.level_indices.flatten()

        if self.is_alignmenet_score:
            self.x_variable = self.ring_luminances
            self.x_variable_indices = self.level_indices

            self.x_gt, self.y_gt = self.get_csf_results()
            self.x_gt = self.x_gt.cpu().numpy()

        self._preview_folder = os.path.join('glare_tests', 'glare_ring_luminance_tests')

    def __len__(self):
        return self.N_contrasts * self.N_levels

    def get_test_condition_parameters(self, index):
        ring_L = self.ring_luminances[self.level_indices[index]]

        if self.is_alignmenet_score:
            multiplier = self.contrasts[self.contrast_indices[index]]
            x_var = self.x_variable[self.x_variable_indices[index]]
            sensitivity_gt = 1 / np.interp(x_var.cpu(), self.x_gt, self.y_gt)
            contrast = 1 / (sensitivity_gt * multiplier)
        else:
            contrast = self.contrasts[self.contrast_indices[index]]

        return contrast, ring_L

    def get_test_condition(self, index):

        contrast, ring_L = self.get_test_condition_parameters(index)

        Lmax = self.display_photometry.get_peak_luminance()

        # Overall background luminance: 0.5 cd/m2
        background_L = 0.5
        # Probe pedestal luminance: 50 cd/m2
        probe_ped_L = 50.0

        RGB_gray = XYZ2RGB709_nd(torch.tensor([0.95047, 1.0, 1.08883], device=self.device).view(1, 1, 3, 1))

        # Convert fixed radii from degrees to pixels
        inner_r_px = self.glare_inner_radius_deg * self.ppd
        outer_r_px = self.glare_outer_radius_deg * self.ppd

        xx, yy = torch.meshgrid(
            torch.linspace(-self.width / 2, self.width / 2, self.width).to(self.device),
            torch.linspace(-self.height / 2, self.height / 2, self.height).to(self.device),
            indexing='xy'
        )
        rr = torch.sqrt(xx**2 + yy**2)

        # Reference: uniform background at 0.5 cd/m2
        reference_condition = torch.ones(self.height, self.width, device=self.device) * background_L

        # Build the ring with specified luminance (absolute cd/m2) - hard edges
        glare = torch.zeros_like(reference_condition)
        ring_mask = (rr > inner_r_px) & (rr <= outer_r_px)
        glare[ring_mask] = ring_L

        reference_condition = (reference_condition + glare).clamp(0, Lmax)

        # Create the probe: a small disk (0.4 deg diameter = 0.2 deg radius) with step edge modulated by Gaussian envelope
        probe_radius_deg = 0.2
        probe_radius_px = probe_radius_deg * self.ppd

        # Center probe at image center
        cx = 0.0
        cy = 0.0

        xx_p = xx - cx
        yy_p = yy - cy
        rr_p = torch.sqrt(xx_p**2 + yy_p**2)

        # Probe region: disk where background is elevated to pedestal luminance
        probe_mask = (rr_p <= probe_radius_px).float()
        probe_bg = probe_mask * (probe_ped_L - background_L)

        # Add probe pedestal to reference (uniform, without edge)
        reference_condition = (reference_condition + probe_bg).clamp(0, Lmax)

        # Gaussian envelope for the edge stimulus
        sigma_px = 0.5 * probe_radius_px
        gauss_env = torch.exp(-(xx_p**2 + yy_p**2) / (2 * sigma_px**2))

        # Step edge: vertical edge with opposite polarity (dark on right, bright on left)
        edge_strength = contrast * probe_ped_L
        edge = torch.where(xx_p > 0, -edge_strength, edge_strength)

        # Modulate by Gaussian envelope
        probe_stimulus = (edge * gauss_env) * probe_mask

        # Combine background and edge stimulus (reference now also has the probe pedestal)
        test_condition = (reference_condition + probe_stimulus).clamp(0, Lmax)

        # Apply color transform to RGB709 space
        reference_condition = reference_condition.view(self.height, self.width, 1, 1) * RGB_gray
        test_condition = test_condition.view(self.height, self.width, 1, 1) * RGB_gray

        vs = video_source.video_source_array(test_condition, reference_condition, 1, dim_order="HWCF", display_photometry=self.display_photometry)
        if self.apply_glare_model:
            vs = GlareModel(vs, self.display_geometry, surround=None, mtf_model=self.mtf_model)

        return vs

    def get_row_header(self):
        return ['Contrast', 'Ring Luminance']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.ring_luminances[self.level_indices]]

    def size(self):
        return self.N_contrasts, self.N_levels

    def units(self):
        return ['log', 'log'], [None, 'cd/m²']

    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        js_command = ""

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()
        units_scales, units_names = self.units()

        y_condition = np.unique(conditions[0].cpu().numpy())
        y_condition_name = condition_names[0]
        if units_names[0] is not None:
            y_condition_name += " [" + units_names[0] + "]"

        x_condition = np.unique(conditions[1].cpu().numpy())
        x_condition_name = condition_names[1]
        if units_names[1] is not None:
            x_condition_name += " [" + units_names[1] + "]"

        z_condition_name = "Metric Prediction"

        predictions = predictions.reshape(self.size())

        x_tick_vals, y_tick_vals = self.get_ticks()

        z = "["
        for i, row in enumerate(predictions):
            if i > 0:
                z += ", "
            z += "["
            for j, el in enumerate(row):
                if j > 0:
                    z += ", "
                if not np.isnan(el):
                    z += str(el)
                else:
                    z += 'null'
            z += "]"
        z += "]"

        x = "["
        for i, el in enumerate(x_condition):
            if i > 0:
                x += ", "
            x += str(el)
        x += "]"

        y = "["
        for i, el in enumerate(y_condition):
            if i > 0:
                y += ", "
            y += str(el)
        y += "]"

        js_command += "var Contourdata = {\n"
        js_command += "z: " + z + ",\nx: " + x + ",\ny: " + y

        extra_params = "type: 'contour',\ncolorscale: 'Viridis',\nncontours: 30,\nline:{width:2},\n"
        if reverse_color_order:
            extra_params += "reversescale: true,\n"

        extra_params += "hovertemplate: '" + x_condition_name + ": %" + r"{x:.3f}" + "<br>" + y_condition_name + ": %" + r"{y:.4f}" + "<br>" + z_condition_name + ": %" + r"{z:.4f}" + "<extra></extra>',\n"
        extra_params += r"contours:{coloring:'lines'}"

        js_command += ",\n" + extra_params
        js_command += "\n};"

        x_gt, y_gt = self.get_csf_results()

        x = "["
        for i, el in enumerate(x_gt):
            if i > 0:
                x += ", "
            x += str(el.cpu().numpy())
        x += "]"

        y = "["
        for i, el in enumerate(y_gt):
            if i > 0:
                y += ", "
            y += str(el)
        y += "]"

        js_command += "\nvar Linedata = {\n"
        js_command += "x: " + x + ",\ny: " + y

        extra_params = "mode: 'lines+markers',\ntype: 'scatter',\nline:{color: 'red', width:2},marker:{symbol:'circle-open', size:8},\nname: 'GlareRingLuminanceGT'"
        js_command += ",\n" + extra_params
        js_command += "\n};"

        js_command += "\nvar data = [Contourdata, Linedata]"

        x = "["
        for i, el in enumerate(x_tick_vals):
            if i > 0:
                x += ", "
            x += str(np.round(el, 2))
        x += "]"

        y = "["
        for i, el in enumerate(y_tick_vals):
            if i > 0:
                y += ", "
            y += str(el)
        y += "]"

        axes_info = "xaxis: {type:'log', range:[" + str(log10(np.min(x_condition))) + "," + str(log10(np.max(x_condition))) + "], tickvals: " + x + ", ticktext: " + x + ", title:'" + x_condition_name + "'}, yaxis: {type:'log', range:[" + str(log10(np.min(y_condition))) + "," + str(log10(np.max(y_condition))) + "], tickvals: " + y + ", ticktext: " + y + ", title:'" + y_condition_name + "'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '" + title + "', y: 0.95}, width:450, height:400, " + axes_info + ",\n" + edge_info + ",\n" + margin_info + "\n};"

        if fig_id:
            js_command += "\nPlotly.newPlot('" + fig_id + "', data, layout);\n\n"

        return js_command

    def get_ticks(self):
        # ticks for ring luminance (x) and contrast (y)
        x = self.ring_luminances.cpu().numpy()
        y = np.logspace(-2, 0, 5)
        return x, y
    
    def get_csf_results(self):
        # This method can be called from parent __init__ before ring_luminances is set.
        if hasattr(self, 'ring_luminances'):
            x = self.ring_luminances
        else:
            x = torch.tensor([0.5, 5.0, 50.0, 500.0, 5000.0], device=self.device)
        y = np.array([0.205118, 0.195109, 0.0680022, 0.248838, 1.42439])
        return x, y

    def get_preview_folder(self):
        return self._preview_folder

    def short_name(self):
        return 'Glare Ring Luminance Test'



class GlareHalfRingLuminanceTest(GlareRingLuminanceTest):

    def __init__(self, apply_glare_model=False, mtf_model='hdrvdp', width=2048, height=1536, N_contrasts=20, N_multiplier=10, is_alignmenet_score=False, device=None):
        super().__init__(apply_glare_model=apply_glare_model, mtf_model=mtf_model, width=width, height=height, N_contrasts=N_contrasts, N_multiplier=N_multiplier, is_alignmenet_score=is_alignmenet_score, device=device)
        self._preview_folder = os.path.join('glare_tests', 'glare_half_ring_luminance_tests')

        if self.is_alignmenet_score:
            self.contrasts = torch.logspace(log10(0.5), log10(2), N_multiplier).to(self.device)
        else:
            self.contrasts = torch.logspace(-2, log10(1), N_contrasts, device=self.device)

    def get_test_condition(self, index):

        contrast, ring_L = self.get_test_condition_parameters(index)

        Lmax = self.display_photometry.get_peak_luminance()

        # Overall background luminance: 0.5 cd/m2
        background_L = 0.5
        # Probe pedestal luminance: 50 cd/m2
        probe_ped_L = 50.0

        RGB_gray = XYZ2RGB709_nd(torch.tensor([0.95047, 1.0, 1.08883], device=self.device).view(1, 1, 3, 1))

        # Convert fixed radii from degrees to pixels
        inner_r_px = self.glare_inner_radius_deg * self.ppd
        outer_r_px = self.glare_outer_radius_deg * self.ppd

        xx, yy = torch.meshgrid(
            torch.linspace(-self.width / 2, self.width / 2, self.width).to(self.device),
            torch.linspace(-self.height / 2, self.height / 2, self.height).to(self.device),
            indexing='xy'
        )
        rr = torch.sqrt(xx**2 + yy**2)

        # Reference: uniform background at 0.5 cd/m2
        reference_condition = torch.ones(self.height, self.width, device=self.device) * background_L

        # Keep only the vertically mirrored half after a 45-degree diagonal cut through the center.
        diagonal_half_plane = yy >= -xx

        # Build the half-ring with specified luminance (absolute cd/m2) - hard edges.
        glare = torch.zeros_like(reference_condition)
        ring_mask = (rr > inner_r_px) & (rr <= outer_r_px) & diagonal_half_plane
        glare[ring_mask] = ring_L

        reference_condition = (reference_condition + glare).clamp(0, Lmax)

        # Create the probe: a small disk (0.4 deg diameter = 0.2 deg radius) with step edge modulated by Gaussian envelope
        probe_radius_deg = 0.2
        probe_radius_px = probe_radius_deg * self.ppd

        # Center probe at image center
        cx = 0.0
        cy = 0.0

        xx_p = xx - cx
        yy_p = yy - cy
        rr_p = torch.sqrt(xx_p**2 + yy_p**2)

        # Probe region: disk where background is elevated to pedestal luminance
        probe_mask = (rr_p <= probe_radius_px).float()
        probe_bg = probe_mask * (probe_ped_L - background_L)

        # Add probe pedestal to reference (uniform, without edge)
        reference_condition = (reference_condition + probe_bg).clamp(0, Lmax)

        # Gaussian envelope for the edge stimulus
        sigma_px = 0.5 * probe_radius_px
        gauss_env = torch.exp(-(xx_p**2 + yy_p**2) / (2 * sigma_px**2))

        # Step edge: vertical edge with opposite polarity (dark on right, bright on left)
        edge_strength = contrast * probe_ped_L
        edge = torch.where(xx_p > 0, -edge_strength, edge_strength)

        # Modulate by Gaussian envelope
        probe_stimulus = (edge * gauss_env) * probe_mask

        # Combine background and edge stimulus (reference now also has the probe pedestal)
        test_condition = (reference_condition + probe_stimulus).clamp(0, Lmax)

        # Apply color transform to RGB709 space
        reference_condition = reference_condition.view(self.height, self.width, 1, 1) * RGB_gray
        test_condition = test_condition.view(self.height, self.width, 1, 1) * RGB_gray

        vs = video_source.video_source_array(test_condition, reference_condition, 1, dim_order="HWCF", display_photometry=self.display_photometry)
        if self.apply_glare_model:
            vs = GlareModel(vs, self.display_geometry, surround=None, mtf_model=self.mtf_model)

        return vs

    def get_csf_results(self):
        x = self.ring_luminances if hasattr(self, 'ring_luminances') else torch.tensor([0.5, 5.0, 50.0, 500.0, 5000.0], device=self.device)
        y = np.array([
            0.211611,  # TODO: add half-ring CSF value for 0.5 cd/m2
            0.16247,  # TODO: add half-ring CSF value for 5.0 cd/m2
            0.144257,  # TODO: add half-ring CSF value for 50.0 cd/m2
            0.312203,  # TODO: add half-ring CSF value for 500.0 cd/m2
            0.789706,  # TODO: add half-ring CSF value for 5000.0 cd/m2
        ])
        return x, y

    def short_name(self):
        return 'Glare Half Ring Luminance Test'




    

