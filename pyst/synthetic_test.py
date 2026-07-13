import numpy as np
from pyst.utils import *
import pyexr
import abc
from PIL import Image
from pycvvdp.video_writer import VideoWriter


class SyntheticTest:
    """
    Abstract class for all synthetic tests
    """

    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

    @abc.abstractmethod
    def get_test_condition_parameters(self, index):
        pass

    @abc.abstractmethod
    def get_test_condition(self, index):
        pass

    @abc.abstractmethod
    def get_condition(self, index):
        pass

    @abc.abstractmethod
    def get_row_header(self):
        pass

    @abc.abstractmethod
    def get_rows_conditions(self):
        pass

    @abc.abstractmethod
    def size(self):
        pass

    @abc.abstractmethod
    def get_ticks(self):
        pass

    @abc.abstractmethod
    def units(self):
        pass

    @abc.abstractmethod
    def short_name(self):
        pass

    @abc.abstractmethod
    def latex_name(self):
        pass

    def compute_alignment_score(self, predictions):
        """
        Default alignment score: Spearman correlation between the swept
        multiplier axis (the first row-header condition) and a metric's
        predictions. Only meaningful when the test was run in alignment-score
        mode (see is_alignmenet_score); returns None otherwise. Test families
        with a different notion of alignment score (e.g. contrast matching)
        override this.
        """
        if not getattr(self, 'is_alignmenet_score', False):
            return None

        from scipy.stats import spearmanr

        multipliers = self.get_rows_conditions()[0]
        try:
            multipliers = multipliers.cpu().numpy()
        except AttributeError:
            pass

        return spearmanr(multipliers, predictions, nan_policy='omit').statistic

    def preview(self, index):
        vid_source = self.get_test_condition(index)

    def save_as_exr(self, index, test_filename, ref_filename):
        vid_source, dm, gm = self.get_condition(index)

        T = vid_source.get_test_frame(0, colorspace='RGB709', device=self.device)
        R = vid_source.get_reference_frame(0, colorspace='RGB709', device=self.device)

        pyexr.write(test_filename, tensor_to_numpy_image(T))
        pyexr.write(ref_filename, tensor_to_numpy_image(R))
    
    def save_as_png(self, index, test_filename, ref_filename):
        vid_source, dm, gm = self.get_condition(index)

        T = Image.fromarray(tensor_to_numpy_image(vid_source.get_test_frame(0, colorspace='display_encoded_dmax', device=self.device)*255).astype(np.uint8))
        R = Image.fromarray(tensor_to_numpy_image(vid_source.get_reference_frame(0, colorspace='display_encoded_dmax', device=self.device)*255).astype(np.uint8))
            
        T.save(test_filename)
        R.save(ref_filename)

    def save_as_video(self, index, test_filename, ref_filename):
        vid_source, dm, gm = self.get_condition(index)

        _, _, N_frames = vid_source.get_video_size()
        fps = vid_source.get_frames_per_second()

        test_vw = VideoWriter(test_filename, fps=fps, codec='h265')
        ref_vw = VideoWriter(ref_filename, fps=fps, codec='h265')

        for ff in range(N_frames):
            T = tensor_to_numpy_image(vid_source.get_test_frame(ff, colorspace='display_encoded_dmax', device=self.device))
            R = tensor_to_numpy_image(vid_source.get_reference_frame(ff, colorspace='display_encoded_dmax', device=self.device))

            test_vw.write_frame_rgb(T)
            ref_vw.write_frame_rgb(R)

        test_vw.close()
        ref_vw.close()

    


        