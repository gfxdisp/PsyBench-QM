import abc

from pycvvdp.video_source import *
from pycvvdp.vq_metric import vq_metric

# A base class for Pytorch Image Quality Metrics

class PyIQM(vq_metric):

    def __init__(self, color_space='display_encoded', device=None):

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        if color_space == 'display_encoded':
            self.color_space = 'display_encoded_dmax'
        else:
            self.color_space = color_space


    def set_display_model(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[]):
        if display_photometry is None:
            self.display_photometry = vvdp_display_photometry.load(display_name, config_paths)
            self.display_name = display_name
        else:
            self.display_photometry = display_photometry
            self.display_name = "unspecified"

        if display_geometry is None:
            self.display_geometry = vvdp_display_geometry.load(display_name, config_paths)
            self.display_name = display_name
        else:
            self.display_geometry = display_geometry