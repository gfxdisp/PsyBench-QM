import torch
from piq import dss
from pycvvdp.vq_metric import vq_metric
from pyst.utils import display_encode

"""
DSS metric. Usage is same as the FovVideoVDP metric (see pytorch_examples).
"""

"""
To be able to use this metric, you will need to install piq: $ pip install piq
"""

class dss_metric(vq_metric):

    def __init__(self, device=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            

        self.color_space = 'Y'

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''

    def predict_video_source(self, vid_source, frame_padding="replicate"):

        _, _, N_frames = vid_source.get_video_size()

        score = 0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2)

            T = display_encode(T, self.display_photometry)
            R = display_encode(R, self.display_photometry)

            score += dss(T, R, data_range=1.) / N_frames

        return score, None

    def short_name(self):
        return "DSS"
    
    @staticmethod
    def name():
        return "DSS"