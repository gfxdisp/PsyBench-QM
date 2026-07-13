from pycvvdp import cvvdp , cvvdp_ml_saliency, cvvdp_ml_transformer 

class cvvdp_met(cvvdp):

    @staticmethod
    def name():
        return 'ColorVideoVDP'
    
    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10

class cvvdp_ml_saliency_met(cvvdp_ml_saliency):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, config_paths=['../metric_calibration/metric_configs/cvvdp_5_6_saliency'])

    @staticmethod
    def name():
        return 'CVVDP-ML-Saliency'

    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10


class cvvdp_ml_transformer_met(cvvdp_ml_transformer):

    @staticmethod
    def name():
        return 'CVVDP-ML-Transformer'

    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10