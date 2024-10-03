try:
    from metrics.pyiqa_metrics.topiq_metric import topiq_metric
    from metrics.pyiqa_metrics.pieapp_metric import pieapp_metric
    from metrics.pyiqa_metrics.nlpd_metric import nlpd_metric
    from metrics.pyiqa_metrics.dists_metric import dists_metric
    from metrics.pyiqa_metrics.ckdn_metric import ckdn_metric
    from metrics.pyiqa_metrics.ahiq_metric import ahiq_metric
    from metrics.pyiqa_metrics.mad_metric import mad_metric
    from metrics.pyiqa_metrics.psnr_metric import psnr_metric
    from metrics.pyiqa_metrics.wadiqam_metric import wadiqam_metric
    from metrics.pyiqa_metrics.qalign_metric import qalign_metric
    from metrics.pyiqa_metrics.liqe_metric import liqe_metric
except ImportError as e:
    import logging
    logging.warning( "Failed to import PYIQA metrics. Run `pip install pyiqa`.")
