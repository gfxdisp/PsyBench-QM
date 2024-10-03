try:
    from metrics.piq_metrics.ssim_metric import ssim_metric
    from metrics.piq_metrics.dss_metric import dss_metric
    from metrics.piq_metrics.fsim_metric import fsim_metric
    from metrics.piq_metrics.gmsd_metric import gmsd_metric
    from metrics.piq_metrics.haarpsi_metric import haarpsi_metric
    from metrics.piq_metrics.iw_ssim_metric import iw_ssim_metric
    from metrics.piq_metrics.mdsi_metric import mdsi_metric
    from metrics.piq_metrics.ms_gmsd_metric import ms_gmsd_metric
    from metrics.piq_metrics.ms_ssim_metric import ms_ssim_metric
    from metrics.piq_metrics.vif_metric import vif_metric
    from metrics.piq_metrics.vsi_metric import vsi_metric
except ImportError as e:
    import logging
    logging.warning( "Failed to import PIQ metrics. Run `pip install piq`.")
