from math import log10
import abc
from pyst.synthetic_test import SyntheticTest
import numpy as np
import os
import matplotlib.pyplot as plt
from pyst.utils import *
import torch 
from pycvvdp.video_source import video_source_array
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry


class GaborPatchTest(SyntheticTest):
    """
    Abstract class for all synthetic tests relying on gabor patches
    """

    # For now we may not need these informations! It is not as important

    def __init__(self, width, height, channel, display_photometry, display_geometry, frames, fps, contrasts, spatial_frequencies=2, temporal_frequencies=0,
                 bkg_luminances=21.4, orientations=0, sizes=2, eccentricities=0, device=None):


        super().__init__(device)

        self.width = width
        self.height = height
        self.channel = channel

        if channel == 'Ach':
            self.dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)
        elif channel == 'RG':
            self.dkl_col_direction = torch.tensor([0, 0.610649, 0], device=self.device)
        elif channel == 'YV':
            self.dkl_col_direction = torch.tensor([0, 0, 4.203636], device=self.device)
        
        self.display_photometry = display_photometry
        self.display_geometry = display_geometry

        self.ppd = display_geometry.get_ppd()
        self.frames = frames
        self.fps = fps

        self.contrasts = contrasts
        self.spatial_frequencies = torch.tensor([spatial_frequencies], device=self.device) if isinstance(spatial_frequencies, (int, float)) else spatial_frequencies.to(self.device)
        self.temporal_frequencies = torch.tensor([temporal_frequencies], device=self.device) if isinstance(temporal_frequencies, (int, float)) else temporal_frequencies.to(self.device)
        self.bkg_luminances = torch.tensor([bkg_luminances], device=self.device) if isinstance(bkg_luminances, (int, float)) else bkg_luminances.to(self.device)
        self.orientations = torch.tensor([orientations], device=self.device) if isinstance(orientations, (int, float)) else orientations.to(self.device)
        self.sizes = torch.tensor([sizes], device=self.device) if isinstance(sizes, (int, float)) else sizes.to(self.device)
        self.eccentricities = torch.tensor([eccentricities], device=self.device) if isinstance(eccentricities, (int, float)) else eccentricities.to(self.device)

        self.N_contrasts = len(self.contrasts)
        self.N_s_freq = len(self.spatial_frequencies)
        self.N_t_freq = len(self.temporal_frequencies)
        self.N_L = len(self.bkg_luminances)
        self.N_orien = len(self.orientations)
        self.N_size = len(self.sizes)
        self.N_ecc = len(self.eccentricities)

        self.contrast_indices, self.s_freq_indices, self.t_freq_indices, self.L_indices, self.orien_indices, self.size_indices, self.ecc_indices = torch.meshgrid(torch.arange(self.N_contrasts), torch.arange(self.N_s_freq), torch.arange(self.N_t_freq), torch.arange(self.N_L), torch.arange(self.N_orien), torch.arange(self.N_size), torch.arange(self.N_ecc), indexing='ij')

        self.contrast_indices = self.contrast_indices.flatten()
        self.s_freq_indices = self.s_freq_indices.flatten()
        self.t_freq_indices = self.t_freq_indices.flatten()
        self.L_indices = self.L_indices.flatten()
        self.orien_indices = self.orien_indices.flatten()
        self.size_indices = self.size_indices.flatten()
        self.ecc_indices = self.ecc_indices.flatten()

        self._preview_folder = 'gabor_tests'
        self._preview_as = None

        self.csf_data_folder = 'pyst/pyst_data/gabor_tests_castlecsf_results'


    def __len__(self):
        return self.N_contrasts*self.N_s_freq*self.N_t_freq*self.N_L*self.N_orien*self.N_size*self.N_ecc

    @abc.abstractmethod
    def get_csf_results(self):
        pass

    @abc.abstractmethod
    def get_ticks(self):
        pass

    def get_test_condition_parameters(self, index):

        contrast = self.contrasts[self.contrast_indices[index]]
        s_freq = self.spatial_frequencies[self.s_freq_indices[index]]
        t_freq = self.temporal_frequencies[self.t_freq_indices[index]]
        L = self.bkg_luminances[self.L_indices[index]]
        orien = self.orientations[self.orien_indices[index]]
        size = self.sizes[self.size_indices[index]]
        ecc = self.eccentricities[self.ecc_indices[index]]

        return contrast, s_freq, t_freq, L, orien, size, ecc
    
    def get_test_condition(self, index):

        contrast, s_freq, t_freq, L, orien, size, ecc = self.get_test_condition_parameters(index)
        Lmax = self.display_photometry.get_peak_luminance()

        XYZ_test, XYZ_reference = generate_gabor_patch(self.width, self.height, self.frames, self.ppd, self.fps, contrast, s_freq, t_freq, orien, L, self.dkl_col_direction, size, ecc, device=self.device)

        # If luminance test we go from XYZ to RGB2020 and then apply appropriate endoing
        # If other tests, we go from XYZ to RGB709, then we apply lin2srgb

        if self.display_photometry.EOTF == 'sRGB':
            test_condition = lin2srgb(XYZ2RGB709_nd(XYZ_test / Lmax).unsqueeze(0))
            reference_condition = lin2srgb(XYZ2RGB709_nd(XYZ_reference / Lmax).unsqueeze(0))

        elif self.display_photometry.EOTF in ['PQ', 'HLG', 'linear']:
            test_condition = XYZ2RGB2020_nd(XYZ_test).unsqueeze(0)
            reference_condition = XYZ2RGB2020_nd(XYZ_reference).unsqueeze(0)

            if self.display_photometry.EOTF == 'PQ':
                test_condition = lin2pq(test_condition)
                reference_condition = lin2pq(reference_condition)
            
            elif self.display_photometry.EOTF == 'HLG':
                test_condition = lin2hlg(test_condition, Lmax)
                reference_condition = lin2hlg(reference_condition, Lmax)

        return video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW", display_photometry=self.display_photometry)

    def plot(self, predictions, output_filename=None, title=None):

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()

        y_condition = conditions[0].reshape(self.size()).cpu().numpy()
        y_condition_name = condition_names[0]

        x_condition = conditions[1].reshape(self.size()).cpu().numpy()
        x_condition_name = condition_names[1]

        predictions = predictions.reshape(self.size())

        fig = plt.figure(figsize=(4, 4))
        plt.contour(x_condition, y_condition, predictions, 30)

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

    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as

    def short_name(self):
        return 'Gabor Patch Test'


class GaborPatchSpatialFreqAchTest(GaborPatchTest):

    def __init__(self, width=1920, height=1080, N_contrasts=20, N_freq=20, device=None):
        
        channel = 'Ach'
        frames = 1
        fps = 1

        spatial_frequencies = torch.logspace(-1, 5, N_freq, base=2)
        contrasts = torch.logspace(log10(0.0035), 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, spatial_frequencies, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'spatial_frequencies_ach')
        self._preview_as = 'image'

    def get_csf_results(self):
        x = torch.logspace(-1, 5, 100, base=2)
        y = np.loadtxt(os.path.join(self.csf_data_folder, 'spatial_freq_ach.txt'))

        return x, y

    def get_ticks(self):
        x = torch.logspace(-1, 5, 7, base=2)
        y = self.contrasts

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Spatial Frequency']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.spatial_frequencies[self.s_freq_indices]]

    def size(self):
        return self.N_contrasts, self.N_s_freq

    def units(self):
        return ['log', 'log'], [None, 'cpd']

    def short_name(self):
        return 'Gabor Patch Spatial Frequency Achromatic Test'

class GaborPatchSpatialFreqRGTest(GaborPatchTest):

    def __init__(self, width=1920, height=1080, N_contrasts=20, N_freq=20, device=None):
        
        channel = 'RG'
        frames = 1
        fps = 1

        spatial_frequencies = torch.logspace(-1, 5, N_freq, base=2)
        contrasts = torch.logspace(log10(0.002), log10(0.2), N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, spatial_frequencies, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'spatial_frequencies_rg')
        self._preview_as = 'image'
    
    def get_csf_results(self):
        x = torch.logspace(-1, 5, 100, base=2)
        y = np.loadtxt(os.path.join(self.csf_data_folder, 'spatial_freq_rg.txt'))

        return x, y

    def get_ticks(self):
        x = torch.logspace(-1, 5, 7, base=2)
        y = self.contrasts

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Spatial Frequency']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.spatial_frequencies[self.s_freq_indices]]

    def size(self):
        return self.N_contrasts, self.N_s_freq

    def units(self):
        return ['log', 'log'], [None, 'cpd']

    def short_name(self):
        return 'Gabor Patch Spatial Frequency RG Test'

class GaborPatchSpatialFreqYVTest(GaborPatchTest):

    def __init__(self, width=1920, height=1080, N_contrasts=20, N_freq=20, device=None):
        
        channel = 'YV'
        frames = 1
        fps = 1

        spatial_frequencies = torch.logspace(-1, 5, N_freq, base=2)
        contrasts = torch.logspace(-2, log10(0.2), N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, spatial_frequencies, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'spatial_frequencies_yv')
        self._preview_as = 'image'

    def get_csf_results(self):
        x = x = torch.logspace(-1, 5, 100, base=2)
        y = np.loadtxt(os.path.join(self.csf_data_folder, 'spatial_freq_yv.txt'))

        return x, y

    def get_ticks(self):
        x = torch.logspace(-1, 5, 7, base=2)
        y = self.contrasts

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Spatial Frequency']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.spatial_frequencies[self.s_freq_indices]]

    def size(self):
        return self.N_contrasts, self.N_s_freq

    def units(self):
        return ['log', 'log'], [None, 'cpd']

    def short_name(self):
        return 'Gabor Patch Spatial Frequency YV Test'


class GaborPatchLuminancePU21AchTest(GaborPatchTest):

    def __init__(self, width=1920, height=1080, N_contrasts=20, N_L=20, device=None):
        
        channel = 'Ach'
        frames = 1
        fps = 1

        bkg_luminances = torch.logspace(-2, 4, N_L)
        contrasts = torch.logspace(log10(0.003), 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=10000, contrast=1000000, source_colorspace='BT.2020-linear')
        gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, bkg_luminances=bkg_luminances, spatial_frequencies=2, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'luminances_pu21_ach')
        self._preview_as = 'image'
    
    def get_csf_results(self):
        x = torch.logspace(-2, 4, 100)
        y = np.loadtxt(os.path.join(self.csf_data_folder, 'luminance_ach.txt'))

        return x, y
    
    def get_ticks(self):
        x = torch.logspace(-2, 4, 7)
        y = self.contrasts

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Luminance']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.bkg_luminances[self.L_indices]]

    def size(self):
        return self.N_contrasts, self.N_L

    def units(self):
        return ['log', 'log'], [None, 'cd/m$^2$']

    def short_name(self):
        return 'Gabor Patch Background Luminance with PU21 encoding - Achromatic Test'


class GaborPatchLuminancePQAchTest(GaborPatchTest):

    def __init__(self, width=1920, height=1080, N_contrasts=20, N_L=20, device=None):
        
        channel = 'Ach'
        frames = 1
        fps = 1

        bkg_luminances = torch.logspace(-2, 4, N_L)
        contrasts = torch.logspace(log10(0.003), 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=10000, contrast=1000000, source_colorspace='BT.2020-PQ')
        gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, bkg_luminances=bkg_luminances, spatial_frequencies=2, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'luminances_pq_ach')
        self._preview_as = 'image'
    
    def get_csf_results(self):
        x = torch.logspace(-2, 4, 100)
        y = np.loadtxt(os.path.join(self.csf_data_folder, 'luminance_ach.txt'))

        return x, y
    
    def get_ticks(self):
        x = torch.logspace(-2, 4, 7)
        y = self.contrasts

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Luminance']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.bkg_luminances[self.L_indices]]

    def size(self):
        return self.N_contrasts, self.N_L

    def units(self):
        return ['log', 'log'], [None, 'cd/m$^2$']

    def short_name(self):
        return 'Gabor Patch Background Luminance with PQ encoding - Achromatic Test'


class GaborPatchLuminanceHLGAchTest(GaborPatchTest):

    def __init__(self, width=1920, height=1080, N_contrasts=20, N_L=20, device=None):
        
        channel = 'Ach'
        frames = 1
        fps = 1

        bkg_luminances = torch.logspace(-2, 4, N_L)
        contrasts = torch.logspace(log10(0.003), 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=10000, contrast=1000000, source_colorspace='BT.2020-HLG')
        gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, bkg_luminances=bkg_luminances, spatial_frequencies=2, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'luminances_hlg_ach')
        self._preview_as = 'image'
    
    def get_csf_results(self):
        x = torch.logspace(-2, 4, 100)
        y = np.loadtxt(os.path.join(self.csf_data_folder, 'luminance_ach.txt'))

        return x, y
    
    def get_ticks(self):
        x = torch.logspace(-2, 4, 7)
        y = self.contrasts

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Luminance']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.bkg_luminances[self.L_indices]]

    def size(self):
        return self.N_contrasts, self.N_L

    def units(self):
        return ['log', 'log'], [None, 'cd/m$^2$']

    def short_name(self):
        return 'Gabor Patch Background Luminance with HLG encoding - Achromatic Test'


# class GaborPatchSizeAchTest(GaborPatchTest):

#     def __init__(self, width=1920, height=1080, N_contrasts=20, N_size=20, device=None):
#         channel = 'Ach'
#         frames = 1
#         fps = 1

#         sizes = torch.logspace(-1, 5, N_size, base=2)
#         contrasts = torch.logspace(-3, 0, N_contrasts)

#         dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
#         gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

#         super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, sizes=sizes, device=device)

#         self._preview_folder = os.path.join(self._preview_folder, 'sizes_ach')
#         self._preview_as = 'image'
    
#     def get_csf_results(self):
#         x = torch.logspace(-1, 5, 100, base=2)
#         y = np.loadtxt(os.path.join(self.csf_data_folder, 'size_ach.txt'))

#         return x, y

#     def get_ticks(self):
#         x = torch.logspace(-1, 5, 7, base=2)
#         y = self.contrasts

#         return x, y

#     def get_row_header(self):
#         return ['Contrast', 'Size']

#     def get_rows_conditions(self):
#         return [self.contrasts[self.contrast_indices], self.sizes[self.size_indices]]

#     def size(self):
#         return self.N_contrasts, self.N_size

#     def units(self):
#         return ['log', 'log'], [None, 'deg$^2$']

#     def short_name(self):
#         return 'Gabor Patch Size Achromatic Test'


# class GaborPatchTempFreqAchTest(GaborPatchTest):

#     def __init__(self, width=1920, height=1080, N_contrasts=20, N_freq=20, device=None):

#         channel = 'Ach'
#         frames = 60
#         fps = 60

#         frequencies = torch.linspace(0, 60, N_freq)
#         contrasts = torch.logspace(-3, 0, N_contrasts)

#         dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
#         gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

#         super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, temporal_frequencies=frequencies, device=device)

#         self._preview_folder = os.path.join(self._preview_folder, 'temporal_frequencies_ach')
#         self._preview_as = 'video'
    
#     def get_csf_results(self):
#         x = torch.linspace(0, 60, 100)
#         y = np.loadtxt(os.path.join(self.csf_data_folder, 'temporal_freq_ach.txt'))

#         return x, y
    
#     def get_ticks(self):
#         x = torch.linspace(0, 60, 7)
#         y = self.contrasts

#         return x, y

#     def get_row_header(self):
#         return ['Contrast', 'Temporal Frequency']

#     def get_rows_conditions(self):
#         return [self.contrasts[self.contrast_indices], self.temporal_frequencies[self.t_freq_indices]]

#     def size(self):
#         return self.N_contrasts, self.N_t_freq

#     def units(self):
#         return ['log', 'linear'], [None, 'Hz']

#     def short_name(self):
#         return 'Gabor Patch Temporal Frequency Achromatic Test'
    

# class GaborPatchEccAchTest(GaborPatchTest):

#     def __init__(self, width=1024, height=1024, N_contrasts=10, N_ecc=7, device=None):

#         channel = 'Ach'
#         frames = 1
#         fps = 1

#         eccentricities = torch.linspace(0, 6, N_ecc)
#         contrasts = torch.logspace(-2, 0, N_contrasts)

#         dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
#         gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

#         super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, eccentricities=eccentricities, sizes=1, device=device)

#         self._preview_folder = os.path.join(self._preview_folder, 'eccentricities_ach')
#         self._preview_as = 'image'

#     def get_row_header(self):
#         return ['Contrast', 'Eccentricity']

#     def get_rows_conditions(self):
#         return [self.contrasts[self.contrast_indices], self.eccentricities[self.ecc_indices]]

#     def size(self):
#         return self.N_contrasts, self.N_ecc

#     def units(self):
#         return ['log', 'linear'], [None, 'deg']

#     def short_name(self):
#         return 'Gabor Patch Eccentricity Achromatic Test'

# class GaborPatchOrientationAchTest(GaborPatchTest):

#     def __init__(self, width=1024, height=1024, N_contrasts=10, N_oriantation=6, device=None):

#         channel = 'Ach'
#         frames = 1
#         fps = 1

#         orientations = torch.linspace(0, 160, N_oriantation)
#         contrasts = torch.logspace(-2, 0, N_contrasts)

#         dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
#         gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

#         super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, orientations=orientations, device=device)

#         self._preview_folder = os.path.join(self._preview_folder, 'orientations_ach')
#         self._preview_as = 'image'

#     def get_row_header(self):
#         return ['Contrast', 'Orientation']

#     def get_rows_conditions(self):
#         return [self.contrasts[self.contrast_indices], self.orientations[self.orien_indices]]

#     def size(self):
#         return self.N_contrasts, self.N_orien

#     def units(self):
#         return ['log', 'linear'], [None, '$^\circ$']

#     def short_name(self):
#         return 'Gabor Patch Orientation Achromatic Test'