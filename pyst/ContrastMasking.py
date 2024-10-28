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
import pandas as pd

class ContrastMakingTest(SyntheticTest):
    """
    Abstract class for all synthetic tests relying on a gabor masker!
    """

    # For now we may not need these informations! It is not as important

    def __init__(self, width, height, test_channel, masker_channel, display_photometry, display_geometry, frames, fps, test_contrasts, masker_contrasts, spatial_frequencies=2, temporal_frequencies=0,
                 bkg_luminances=21.4, masker_orientations=0, test_orientations=0, test_sizes=1, device=None):


        super().__init__(device)

        self.width = width
        self.height = height

        self.test_channel = test_channel
        self.masker_channel = masker_channel

        if test_channel == 'Ach':
            self.test_dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)
        elif test_channel == 'RG':
            self.test_dkl_col_direction = torch.tensor([0, 0.610649, 0], device=self.device)
        elif test_channel == 'YV':
            self.test_dkl_col_direction = torch.tensor([0, 0, 4.203636], device=self.device)
        
        if masker_channel == 'Ach':
            self.masker_dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)
        elif masker_channel == 'RG':
            self.masker_dkl_col_direction = torch.tensor([0, 0.610649, 0], device=self.device)
        elif masker_channel == 'YV':
            self.masker_dkl_col_direction = torch.tensor([0, 0, 4.203636], device=self.device)
        
        self.display_photometry = display_photometry
        self.display_geometry = display_geometry

        self.ppd = display_geometry.get_ppd()
        self.frames = frames
        self.fps = fps

        self.test_contrasts = test_contrasts
        self.masker_contrasts = masker_contrasts

        self.spatial_frequencies =  torch.tensor([spatial_frequencies], device=self.device) if isinstance(spatial_frequencies, (int, float)) else spatial_frequencies.to(self.device)
        self.temporal_frequencies = torch.tensor([temporal_frequencies], device=self.device) if isinstance(temporal_frequencies, (int, float)) else temporal_frequencies.to(self.device)
        self.bkg_luminances = torch.tensor([bkg_luminances], device=self.device) if isinstance(bkg_luminances, (int, float)) else bkg_luminances.to(self.device)

        self.masker_orientations = torch.tensor([masker_orientations], device=self.device) if isinstance(masker_orientations, (int, float)) else masker_orientations.to(self.device)
        self.test_orientations = torch.tensor([test_orientations], device=self.device) if isinstance(test_orientations, (int, float)) else test_orientations.to(self.device)

        self.test_sizes = torch.tensor([test_sizes], device=self.device) if isinstance(test_sizes, (int, float)) else test_sizes.to(self.device)

        self.N_test_contrasts = len(self.test_contrasts)
        self.N_masker_contrasts = len(self.masker_contrasts)
        self.N_s_freq = len(self.spatial_frequencies)
        self.N_t_freq = len(self.temporal_frequencies)
        self.N_L = len(self.bkg_luminances)
        self.N_test_orien = len(self.test_orientations)
        self.N_masker_orien = len(self.masker_orientations)
        self.N_size = len(self.test_sizes)

        self.test_contrast_indices, self.masker_contrast_indices, self.s_freq_indices, self.t_freq_indices, self.L_indices, self.test_orien_indices, self.masker_orien_indices, self.size_indices = torch.meshgrid(torch.arange(self.N_test_contrasts), torch.arange(self.N_masker_contrasts), torch.arange(self.N_s_freq), torch.arange(self.N_t_freq), torch.arange(self.N_L), torch.arange(self.N_test_orien), torch.arange(self.N_masker_orien), torch.arange(self.N_size), indexing='ij')

        self.test_contrast_indices = self.test_contrast_indices.flatten()
        self.masker_contrast_indices = self.masker_contrast_indices.flatten()
        self.s_freq_indices = self.s_freq_indices.flatten()
        self.t_freq_indices = self.t_freq_indices.flatten()
        self.L_indices = self.L_indices.flatten()
        self.test_orien_indices = self.test_orien_indices.flatten()
        self.masker_orien_indices = self.masker_orien_indices.flatten()
        self.size_indices = self.size_indices.flatten()

        self._preview_folder = 'contrast_masking'
        self._preview_as = None

        self.data_folder = 'pyst/pyst_data/contrast_masking'


    def __len__(self):
        return self.N_test_contrasts*self.N_masker_contrasts*self.N_s_freq*self.N_t_freq*self.N_L*self.N_test_orien*self.N_masker_orien*self.N_size

    @abc.abstractmethod
    def get_contrast_masking_results(self):
        pass

    def get_test_condition_parameters(self, index):

        test_contrast = self.test_contrasts[self.test_contrast_indices[index]]
        masker_contrast = self.masker_contrasts[self.masker_contrast_indices[index]]

        s_freq = self.spatial_frequencies[self.s_freq_indices[index]]
        t_freq = self.temporal_frequencies[self.t_freq_indices[index]]
        L = self.bkg_luminances[self.L_indices[index]]

        test_orien = self.test_orientations[self.test_orien_indices[index]]
        masker_orien = self.masker_orientations[self.masker_orien_indices[index]]

        size = self.test_sizes[self.size_indices[index]]

        return test_contrast, masker_contrast, s_freq, t_freq, L, test_orien, masker_orien, size
    
    def get_test_condition(self, index):

        ecc = 0

        test_contrast, masker_contrast, s_freq, t_freq, L, test_orien, masker_orien, size = self.get_test_condition_parameters(index)
        Lmax = self.display_photometry.get_peak_luminance()

        # We need to calculate both LMS mean and LMS delta, depending on the channel!

        # Yxy_bkg = torch.cat((L.view(1), self.xy_bkg), axis=0).reshape((1, 3))
        
        # LMS_mean_test = torch.tensor([0, 0, 0], device=self.device)
        # LMS_mean_masker = XYZ2LMS2006(Yxy2XYZ(Yxy_bkg))

        # LMS_delta_test = DKL2LMS2006_D65(self.test_dkl_col_direction) * L * test_contrast
        # LMS_delta_masker = DKL2LMS2006_D65(self.masker_dkl_col_direction) * L * masker_contrast

        # LMS_delta_test = LMS_mean_masker * test_contrast
        # LMS_delta_masker = LMS_mean_masker * masker_contrast

        XYZ_target, XYZ_masker = generate_gabor_mask(self.width, self.height, self.ppd, test_contrast, masker_contrast, s_freq, test_orien, masker_orien, L, self.test_dkl_col_direction, self.masker_dkl_col_direction, size, device=self.device)

        test_patch = XYZ2RGB709_nd(XYZ_target / Lmax).unsqueeze(0)
        masker = XYZ2RGB709_nd(XYZ_masker / Lmax).unsqueeze(0)

        test_condition = lin2srgb(test_patch + masker)
        reference_condition = lin2srgb(masker)

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

        formatted_xticks = [f'{tick:.2g}' for tick in np.unique(x_condition)[::3]]
        formatted_yticks = [f'{tick:.2g}' for tick in np.unique(y_condition)[::3]]

        plt.xticks(np.unique(x_condition)[::3], formatted_xticks, fontsize=10)
        plt.yticks(np.unique(y_condition)[::3], formatted_yticks, fontsize=10)

        # Plot the csf results

        x_gt, y_gt = self.get_contrast_masking_results()

        plt.plot(x_gt, y_gt, marker='o', color='red', linewidth=1, markerfacecolor='none', markeredgecolor='red')

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
        return 'Contrast Masking Test'


class ContrastMakingAchTest(ContrastMakingTest):

    def __init__(self, width=1920, height=1080, N_test_contrasts=20, N_masker_contrasts=20, device=None):
        
        test_channel = 'Ach'
        masker_channel = 'Ach'

        frames = 1
        fps = 1

        test_contrasts = torch.logspace(np.log10(0.005), np.log10(0.4), N_test_contrasts)
        masker_contrasts = torch.logspace(np.log10(0.005), np.log10(0.4), N_masker_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([1920, 1080], diagonal_size_inches=30, ppd=64)

        super().__init__(width, height, test_channel, masker_channel, dm, gm, frames, fps, test_contrasts, masker_contrasts, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'ach')
        self._preview_as = 'image'

    def get_contrast_masking_results(self):

        df = pd.read_csv(os.path.join(self.data_folder, 'foley_data.csv'))
        df = df[(df['orientation'] == 0) & (df['c_mask'] != 0)]
        x = 10 ** (df['c_mask'].to_numpy()/20)
        y = 10 ** (df['c_target'].to_numpy()/20)
        
        return x, y

    def get_row_header(self):
        return ['Test Contrast', 'Masker Contrast']

    def get_rows_conditions(self):
        return [self.test_contrasts[self.test_contrast_indices], self.masker_contrasts[self.masker_contrast_indices]]

    def size(self):
        return self.N_test_contrasts, self.N_masker_contrasts

    def units(self):
        return ['log', 'log'], [None, None]

    def short_name(self):
        return 'Contrast Masking (Gabor Patch) Achromatic Test'

