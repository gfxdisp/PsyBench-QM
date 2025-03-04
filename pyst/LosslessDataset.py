import abc
import os
import torch
import torch.utils.data as D
from pycvvdp.video_source_file import load_image_as_array
from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import spearmanr
import pandas as pd


class LosslessDataset(D.Dataset, metaclass=abc.ABCMeta):

    def __init__(self, path=os.path.join('quality_dataset_tests', 'vis_lossess_view_cond', 'images'), device=None):
        super().__init__()
        self.path = path

        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # gt_path = os.path.join('quality_dataset_tests', 'x_distortion_levels.json')
        # with open(gt_path, 'r') as fp:
        #     self.data = json.load(fp)

        # self.probabilities = torch.tensor([0.25, 0.5, 0.75], device=self.device)

        gt_path = os.path.join('quality_dataset_tests', 'vis_lossess_view_cond', 'data_averaged', 'vlt_clean.csv')
        self.data = pd.read_csv(gt_path)

        self.probabilities = torch.tensor([0.25], device=self.device)
        self.images = torch.linspace(0, 19, 20, device=self.device)

        self.N_prob = len(self.probabilities)
        self.N_images = len(self.images)

        self.prob_indices, self.images_indices = torch.meshgrid(torch.arange(self.N_prob), torch.arange(self.N_images), indexing='ij')

        self.prob_indices = self.prob_indices.flatten()
        self.images_indices = self.images_indices.flatten()

        self.display_photometry = vvdp_display_photo_eotf(Y_peak=220, contrast=1000, source_colorspace='sRGB')
        self.display_geometry = vvdp_display_geometry([512, 512], diagonal_size_inches=30, ppd=60)

    def __getitem__(self, index):
        """
        Returns:
            vs:         video source object
            disp_photo: photometric display model
            disp_geom:  geometric display model
        """

        assert index in range(self.__len__()), f'{index} is out of range, len={self.__len__()}'

        prob = self.probabilities[self.prob_indices[index]].item()
        im = int(self.images[self.images_indices[index]].item())

        if prob == 0.25:
            index = 0
        elif prob == 0.5:
            index = 1
        elif prob == 0.75:
            index = 2

        if im < 10:
            im_name = 'i'+str(im)+'webp'
        else:
            im_name = 'i'+str(im)+'jpeg'

        #distortion_level = int(self.data[im_name]['Lum_220_PPD_60'][index])

        all_possible_dis = np.linspace(0, 100, 51)
        distortion_level = self.data['q_220_60'][im]
        distortion_level = int(all_possible_dis[np.argmin(np.abs(distortion_level - all_possible_dis))])

        ref_path = os.path.join(self.path, im_name+'_ref.png')
        test_path = os.path.join(self.path, im_name+'_'+str(distortion_level)+'.png')

        reference_cont = load_image_as_array(ref_path)
        test_cont = load_image_as_array(test_path)


        vs = video_source.video_source_array( test_cont, reference_cont, 1, dim_order='HWC', display_photometry=self.display_photometry )

        return vs, self.display_photometry, self.display_geometry


    def __len__(self):
        return self.N_images * self.N_prob

    def size(self):
        return self.N_prob, self.N_images

    def plot(self, predictions, metric_name):
        #error = self.error(predictions)
        fig = plt.figure(figsize=(5, 4))
        predictions = predictions.reshape(self.size())
        for prediction, prob in zip(predictions, self.probabilities):
            plt.plot(self.images.cpu().numpy(), prediction, label='prob: '+str(prob.cpu().item()))

        plt.xlabel('Image Distortion')
        plt.ylabel('Metric Prediction')

        #title = metric_name+' [AS='+str(np.round(error, 3))+']'

        plt.legend()
        #plt.title(title)

        fig.savefig(os.path.join('figs', 'aliaksei_2', metric_name+'.png'))

        plt.close(fig)

    def error(self, predictions):
        prob = self.probabilities[self.prob_indices].cpu().numpy()
        return spearmanr(prob, predictions).statistic



    def get_row_header(self):
        return ['Prob', 'Image']

    def get_rows_conditions(self):
        return [self.probabilities[self.prob_indices], self.images[self.images_indices]]

    def get_preview_folder(self):
        return 'aliaksei_2'

    def short_name(self):
        return 'Lossless Dataset - Test'