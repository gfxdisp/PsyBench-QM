from pycvvdp import cvvdp
from metrics.glare_model import GlareModel

class cvvdp_ncsf_mtf_met(cvvdp):

    def __init__(self, device=None):
        super().__init__(device=device, config_paths=['metrics/cvvdp_ncsf_configs'])
    
    def predict_video_source(self, vid_source):
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=None)
        return super().predict_video_source(vid_source)

    @staticmethod
    def name():
        return 'ColorVideoVDP_nCSF_mtf'
    
    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10

class cvvdp_ncsf_mtf_cie99_met(cvvdp):

    def __init__(self, device=None):
        super().__init__(device=device, config_paths=['metrics/cvvdp_ncsf_cie99_configs'])
    
    def predict_video_source(self, vid_source):
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=None, mtf_model='cie99')
        return super().predict_video_source(vid_source)

    @staticmethod
    def name():
        return 'ColorVideoVDP_nCSF_mtf_cie99'
    
    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10

class cvvdp_ncsf_mtf_deeley_met(cvvdp):

    def __init__(self, device=None):
        super().__init__(device=device, config_paths=['metrics/cvvdp_ncsf_deeley_configs'])

    def predict_video_source(self, vid_source):
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=None, mtf_model='deeley')
        return super().predict_video_source(vid_source)

    @staticmethod
    def name():
        return 'ColorVideoVDP_nCSF_mtf_deeley'
    
    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10

class cvvdp_ncsf_mtf_ijspeert_met(cvvdp):

    def __init__(self, device=None):
        super().__init__(device=device, config_paths=['metrics/cvvdp_ncsf_ijspeert_configs'])

    def predict_video_source(self, vid_source):
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=None, mtf_model='ijspeert')
        return super().predict_video_source(vid_source)

    @staticmethod
    def name():
        return 'ColorVideoVDP_nCSF_mtf_ijspeert'
    
    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10
    
class cvvdp_ncsf_mtf_sensitivity_10_met(cvvdp):

    def __init__(self, device=None):
        super().__init__(device=device, config_paths=['metrics/cvvdp_ncsf_sensitivity_10_configs'])

    def predict_video_source(self, vid_source):
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=None)
        return super().predict_video_source(vid_source)

    @staticmethod
    def name():
        return 'ColorVideoVDP_nCSF_mtf_sensitivity_10'
    
    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10
    
class cvvdp_ncsf_mtf_trained_met(cvvdp):

    def __init__(self, device=None):
        super().__init__(device=device, config_paths=['metrics/cvvdp_ncsf_trained_configs'])
    
    def predict_video_source(self, vid_source):
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=None)
        return super().predict_video_source(vid_source)

    @staticmethod
    def name():
        return 'ColorVideoVDP_nCSF_mtf_trained'
    
    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10