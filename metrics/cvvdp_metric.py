from pycvvdp import cvvdp , cvvdp_ml_saliency, cvvdp_ml_transformer #, cvvdp_ml_transformer_bands, cvvdp_ml_dino_fusion, cvvdp_ml_saliency_plus
#from pycvvdp import cvvdp_ml_dino_fusion_bandfreq
#from python_metrics import dino_met

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


# class cvvdp_ml_saliency_plus_met(cvvdp_ml_saliency_plus):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs, config_paths=['../metric_calibration/metric_configs/cvvdp_5_6_saliency_plus'])

#     @staticmethod
#     def name():
#         return 'CVVDP-ML-Saliency-Plus'

#     @staticmethod
#     def is_lower_better():
#         return False
    
#     @staticmethod
#     def predictions_range():
#         return 0, 10

# class cvvdp_ml_saliency_plus_finetuned_met(cvvdp_ml_saliency_plus):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs, config_paths=['../metric_calibration/metric_configs/cvvdp_5_6_saliency_plus_finetuned'])

#     @staticmethod
#     def name():
#         return 'CVVDP-ML-Saliency-Plus-Finetuned'

#     @staticmethod
#     def is_lower_better():
#         return False
    
#     @staticmethod
#     def predictions_range():
#         return 0, 10

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

    
# class cvvdp_ml_dino_fusion_met(cvvdp_ml_dino_fusion):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs, config_paths=['../metric_calibration/metric_configs/cvvdp_5_6_dino_fusion'])
    
#     @staticmethod
#     def name():
#         return 'CVVDP-ML-DINO-Fusion'

#     @staticmethod
#     def is_lower_better():
#         return False
    
#     @staticmethod
#     def predictions_range():
#         return 0, 10

# class cvvdp_ml_dino_fusion_finetuned_met(cvvdp_ml_dino_fusion):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs, config_paths=['../metric_calibration/metric_configs/cvvdp_5_6_dino_fusion_finetuned'])
    
#     @staticmethod
#     def name():
#         return 'CVVDP-ML-DINO-Fusion-Finetuned'

#     @staticmethod
#     def is_lower_better():
#         return False
    
#     @staticmethod
#     def predictions_range():
#         return 0, 10

# class cvvdp_ml_dino_fusion_bandfreq_met(cvvdp_ml_dino_fusion_bandfreq):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs, config_paths=['../metric_calibration/metric_configs/cvvdp_5_6_dino_fusion_bandfreq'])
    
#     @staticmethod
#     def name():
#         return 'CVVDP-ML-DINO-Fusion-BandFreq'

#     @staticmethod
#     def is_lower_better():
#         return False
    
#     @staticmethod
#     def predictions_range():
#         return 0, 10
    
# class dino_v1_cls_met(dino_met):
    
#     @staticmethod
#     def name():
#         return 'DINO-V1-CLS'

#     @staticmethod
#     def is_lower_better():
#         return False
    
#     @staticmethod
#     def predictions_range():
#         return 0, 10