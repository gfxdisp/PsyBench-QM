from math import log10

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
                 bkg_luminances=21.4, orientations=0, sizes=1.5, eccentricities=0, device=None):


        super().__init__(device)

        self.width = width
        self.height = height
        self.channel = channel

        self.xy_bkg = torch.tensor([0.31271, 0.32902], device=self.device)
        if channel == 'Ach':
            self.xy_modulation = torch.tensor([0.3127, 0.3290], device=self.device)
            self.dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)
        elif channel == 'RG':
            self.xy_modulation = torch.tensor([0.5027, 0.2322], device=self.device)
            self.dkl_col_direction = torch.tensor([0, 0.610649, 0], device=self.device)
        elif channel == 'YV':
            self.xy_modulation = torch.tensor([0.3313, 0.3738], device=self.device)
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


    def __len__(self):
        return self.N_contrasts*self.N_s_freq*self.N_t_freq*self.N_L*self.N_orien*self.N_size*self.N_ecc


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

        # We need to calculate both LMS mean and LMS delta, depending on the channel!

        Yxy_bkg = torch.cat((L.view(1), self.xy_bkg), axis=0).reshape((1, 3))
        LMS_mean = XYZ2LMS2006(Yxy2XYZ(Yxy_bkg))

        # if self.channel == 'Ach':
        #     LMS_delta = LMS_mean * contrast
        # else:
        #     Yxy_modulation = torch.cat((L.view(1), self.xy_modulation), axis=0).reshape((1, 3))
        #     LMS_mdoulation_peak = XYZ2LMS2006(Yxy2XYZ(Yxy_modulation))
        #     LMS_delta = (LMS_mdoulation_peak - LMS_mean) * contrast

        LMS_delta = DKL2LMS2006_D65(self.dkl_col_direction) * L * contrast

        gabor_patch = generate_gabor_patch(self.width, self.height, self.frames, self.ppd, self.fps, s_freq, t_freq, orien, LMS_mean, LMS_delta, size, ecc, device=self.device).permute(2, 3, 0, 1) # FROM HWCF to CFHW

        # If luminance test we go from LMS2006 to RGB2020
        # If other tests, we go from LMS2006 to RGB709, then we apply lin2srgb

        if self.display_photometry.is_input_display_encoded():
            # we go from LMS2006 to RGB709, then we apply lin2srgb
            test_condition = XYZ2RGB709_nd(LMS20062XYZ_nd(gabor_patch)).unsqueeze(0)
            test_condition = lin2srgb(test_condition / Lmax)
            reference_condition = lin2srgb(torch.ones(1, 3, self.frames, self.height, self.width).to(self.device) * L / Lmax)# need to kind of divide by Lpeak and then apply sRGB encoding!

        else:
            # we go from LMS2006 to RGB2020
            test_condition = XYZ2RGB2020_nd(LMS20062XYZ_nd(gabor_patch)).unsqueeze(0)
            reference_condition = torch.ones(1, 3, self.frames, self.height, self.width).to(self.device) * L

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

        formatted_xticks = [f'{tick:g}' for tick in np.unique(x_condition)]
        formatted_yticks = [f'{tick:.2g}' for tick in np.unique(y_condition)]

        plt.xticks(np.unique(x_condition), formatted_xticks, fontsize=10)
        plt.yticks(np.unique(y_condition), formatted_yticks, fontsize=10)

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

    def __init__(self, width=512, height=512, N_contrasts=10, N_freq=7, device=None):
        
        channel = 'Ach'
        frames = 1
        fps = 1

        spatial_frequencies = torch.logspace(-1, 5, N_freq, base=2)
        contrasts = torch.logspace(-2, 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, spatial_frequencies, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'spatial_frequencies_ach')
        self._preview_as = 'image'

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

    def __init__(self, width=512, height=512, N_contrasts=10, N_freq=7, device=None):
        
        channel = 'RG'
        frames = 1
        fps = 1

        spatial_frequencies = torch.logspace(-1, 5, N_freq, base=2)
        contrasts = torch.logspace(-2, log10(0.4), N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, spatial_frequencies, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'spatial_frequencies_rg')
        self._preview_as = 'image'

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

    def __init__(self, width=512, height=512, N_contrasts=10, N_freq=7, device=None):
        
        channel = 'YV'
        frames = 1
        fps = 1

        spatial_frequencies = torch.logspace(-1, 5, N_freq, base=2)
        contrasts = torch.logspace(-2, log10(0.3), N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, spatial_frequencies, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'spatial_frequencies_yv')
        self._preview_as = 'image'

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


class GaborPatchLuminanceAchTest(GaborPatchTest):

    def __init__(self, width=512, height=512, N_contrasts=10, N_L=7, device=None):
        
        channel = 'Ach'
        frames = 1
        fps = 1

        bkg_luminances = torch.logspace(-2, 4, N_L)
        contrasts = torch.logspace(-2, 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=10000, contrast=1000000, source_colorspace='BT.2020-linear')
        gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, bkg_luminances=bkg_luminances, spatial_frequencies=2, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'luminances_ach')
        self._preview_as = 'image'

    def get_row_header(self):
        return ['Contrast', 'Luminance']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.bkg_luminances[self.L_indices]]

    def size(self):
        return self.N_contrasts, self.N_L

    def units(self):
        return ['log', 'log'], [None, 'cd/m$^2$']

    def short_name(self):
        return 'Gabor Patch Background Luminance Achromatic Test'


class GaborPatchSizeAchTest(GaborPatchTest):

    def __init__(self, width=512, height=512, N_contrasts=10, N_size=7, device=None):
        channel = 'Ach'
        frames = 1
        fps = 1

        sizes = torch.logspace(-3, 3, N_size, base=2)
        contrasts = torch.logspace(-2, 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, sizes=sizes, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'sizes_ach')
        self._preview_as = 'image'

    def get_row_header(self):
        return ['Contrast', 'Size']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.sizes[self.size_indices]]

    def size(self):
        return self.N_contrasts, self.N_size

    def units(self):
        return ['log', 'log'], [None, 'deg$^2$']

    def short_name(self):
        return 'Gabor Patch Size Achromatic Test'


class GaborPatchEccAchTest(GaborPatchTest):

    def __init__(self, width=1024, height=1024, N_contrasts=10, N_ecc=7, device=None):

        channel = 'Ach'
        frames = 1
        fps = 1

        eccentricities = torch.linspace(0, 6, N_ecc)
        contrasts = torch.logspace(-2, 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, eccentricities=eccentricities, sizes=1, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'eccentricities_ach')
        self._preview_as = 'image'

    def get_row_header(self):
        return ['Contrast', 'Eccentricity']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.eccentricities[self.ecc_indices]]

    def size(self):
        return self.N_contrasts, self.N_ecc

    def units(self):
        return ['log', 'linear'], [None, 'deg']

    def short_name(self):
        return 'Gabor Patch Eccentricity Achromatic Test'

class GaborPatchOrientationAchTest(GaborPatchTest):

    def __init__(self, width=1024, height=1024, N_contrasts=10, N_oriantation=6, device=None):

        channel = 'Ach'
        frames = 1
        fps = 1

        orientations = torch.linspace(0, 160, N_oriantation)
        contrasts = torch.logspace(-2, 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, orientations=orientations, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'orientations_ach')
        self._preview_as = 'image'

    def get_row_header(self):
        return ['Contrast', 'Orientation']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.orientations[self.orien_indices]]

    def size(self):
        return self.N_contrasts, self.N_orien

    def units(self):
        return ['log', 'linear'], [None, '$^\circ$']

    def short_name(self):
        return 'Gabor Patch Orientation Achromatic Test'

class GaborPatchTempFreqAchTest(GaborPatchTest):

    def __init__(self, width=1024, height=1024, N_contrasts=10, N_freq=7, device=None):

        channel = 'Ach'
        frames = 60
        fps = 60

        frequencies = torch.linspace(0, 60, N_freq)
        contrasts = torch.logspace(-2, 0, N_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([3840, 2160], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, channel, dm, gm, frames, fps, contrasts, temporal_frequencies=frequencies, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'temporal_frequencies_ach')
        self._preview_as = 'video'

    def get_row_header(self):
        return ['Contrast', 'Temporal Frequency']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.temporal_frequencies[self.t_freq_indices]]

    def size(self):
        return self.N_contrasts, self.N_t_freq

    def units(self):
        return ['log', 'linear'], [None, 'Hz']

    def short_name(self):
        return 'Gabor Patch Temporal Frequency Achromatic Test'