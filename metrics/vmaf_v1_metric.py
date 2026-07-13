import json
import numpy as np
import os, os.path as osp
from time import time
from tqdm import trange
import torch
import tempfile
from subprocess import Popen, PIPE


from pycvvdp.vq_metric import vq_metric
from utils.yuv_utils import YUVWriter
# Optional GPU-accelerated writer (falls back to CPU writer if not available)
try:
    from utils.yuv_utils_gpu import YUVWriterGPU
except Exception:
    YUVWriterGPU = None

"""
VMAF metric wrapper.

Dependencies:

Install libvmaf with the vmaf executable from: https://github.com/Netflix/vmaf/tree/master/libvmaf

To install in a custom directory, add `-Dprefix=$HOME/<your_dir>` when running `meson`. 

This metric will need to decode test/reference each video to yuv first. Make sure you have enough space in /tmp, or better create ramdisc and pass the path run run_metric.py:

python run_metric.py ... --metric-cache /media/ramdisc/

To create a ramdisc, execute as root:

path=/media/ramdisc
mkdir -p $path
chmod 777 $path
mount -t tmpfs -o size=16048M tmpfs $path

"""

"""
Example script to compute SDR VMAF scores.

This script demonstrates how to compute VMAF scores for SDR content using
vmafplus_v1.0.16 models.

Usage:
    python run_sdr_vmaf.py --reference <ref.yuv> --distorted <dist.yuv> \
           --width 1920 --height 1080 --pix_fmt yuv420p --model 1080p

Models available:
    - base: vmafplus_v1.0.16_3d0h.json (default, resolution-agnostic)
    - 480p: vmafplus_v1.0.16_3d0h.json_480
    - 720p: vmafplus_v1.0.16_3d0h.json_720
    - 1080p: vmafplus_v1.0.16_3d0h.json_2160 (also works for 1080p)
"""

import argparse
import os
import sys

# Add project paths to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'cpp_metrics/vmaf-research-dounia'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'cpp_metrics/vmaf-research-dounia', 'private/python/src'))

from vmaf.core.asset import Asset
from vmaf_private.core.quality_runner import VmafPlusQualityRunner


def get_model_path(resolution='base'):
    """Get the path to the appropriate VMAF model."""
    models_dir = os.path.join(PROJECT_ROOT, '../metric_calibration', 'cpp_metrics/vmaf-research-dounia', 'models/sdr/vmafplus_v1.0.16')

    model_map = {
        'base': 'vmafplus_v1.0.16_3d0h.json',
        '480p': 'vmafplus_v1.0.16_3d0h.json_480',
        '720p': 'vmafplus_v1.0.16_3d0h.json_720',
        '1080p': 'vmafplus_v1.0.16_3d0h.json_2160',
        '2160p': 'vmafplus_v1.0.16_3d0h.json_2160',
    }

    model_file = model_map.get(resolution, model_map['base'])
    model_path = os.path.join( models_dir, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path


def compute_vmaf(reference_path, distorted_path, width, height, pix_fmt='yuv420p',
                 model_resolution='base', workdir=None):
    """
    Compute VMAF score between reference and distorted videos.

    Args:
        reference_path: Path to reference YUV file
        distorted_path: Path to distorted YUV file
        width: Video width in pixels
        height: Video height in pixels
        pix_fmt: Pixel format (default: yuv420p)
        model_resolution: Model to use (base, 480p, 720p, 1080p)
        workdir: Working directory for temporary files (default: ./vmaf_workdir)

    Returns:
        Dictionary containing VMAF results
    """

    # Verify input files exist
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not os.path.exists(distorted_path):
        raise FileNotFoundError(f"Distorted file not found: {distorted_path}")

    # Get model path
    model_path = get_model_path(model_resolution)
    print(f"Using model: {model_path}")

    # Create asset
    asset = Asset(
        dataset="custom",
        content_id=0,
        asset_id=0,
        ref_path=reference_path,
        dis_path=distorted_path,
        asset_dict={
            'width': width,
            'height': height,
            'yuv_type': pix_fmt,
        }
    )

    # Create quality runner
    runner = VmafPlusQualityRunner(
        [asset],
        None,  # No logger
        fifo_mode=True,
        delete_workdir=True,
        result_store=None,
        optional_dict={'model_filepath': model_path}
    )

    # Run VMAF computation
    runner.run(parallelize=False)

    # Get results
    results = runner.results[0]

    # Extract key metrics
    vmaf_score = np.mean(results.result_dict['VMAFPLUS_scores'])

    return vmaf_score


class vmaf_v1_met(vq_metric):
    def __init__(self, cache_ref_loc=None, device=None):
 
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.cache_ref_loc = cache_ref_loc

 
    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, frame_padding="replicate"):

        # T_vid and R_vid are the tensors of the size (1,1,N,H,W)
        # where:
        # N - the number of frames
        # H - height in pixels
        # W - width in pixels
        # Both images must contain linear absolute luminance values in cd/m^2
        # 
        # We assume the pytorch default NCDHW layout

        h, w, N_frames = vid_source.get_video_size()

        with tempfile.TemporaryDirectory(prefix=self.short_name(), dir=self.cache_ref_loc) as temp_dir:

            vprops = dict()
            vprops["width"]=w
            vprops["height"]=h
            vprops["fps"] = vid_source.get_frames_per_second()
            vprops["bit_depth"] = 10
            vprops["color_space"] = '709'
            vprops["chroma_ss"] = '444'

            # choose GPU writer when possible
            use_gpu_writer = (hasattr(self, 'device') and getattr(self.device, 'type', None) == 'cuda' and YUVWriterGPU is not None)

            if use_gpu_writer:
                test_writer = YUVWriterGPU(os.path.join(temp_dir, "test"), vprops, device=self.device)
                ref_writer = YUVWriterGPU(os.path.join(temp_dir, "ref"), vprops, device=self.device)
            else:
                test_writer = YUVWriter(os.path.join(temp_dir, "test"), vprops)
                ref_writer = YUVWriter(os.path.join(temp_dir, "ref"), vprops)

            with test_writer as test_vw, ref_writer as ref_vw:

                for ff in range(N_frames):
                    # keep tensors on GPU when using the GPU writer
                    T = vid_source.get_test_frame(ff, device=self.device, colorspace='display_encoded_01').squeeze().permute(1,2,0)
                    R = vid_source.get_reference_frame(ff, device=self.device, colorspace='display_encoded_01').squeeze().permute(1,2,0)

                    if not use_gpu_writer:
                        T = T.cpu().numpy()
                        R = R.cpu().numpy()
                    
                    test_vw.append_frame_rgb(T)
                    ref_vw.append_frame_rgb(R)

                quality = compute_vmaf(
                    reference_path=ref_vw.fname,
                    distorted_path=test_vw.fname,
                    width=w,
                    height=h,
                    pix_fmt='yuv444p10le',
                    model_resolution='1080p',
                    workdir=None
                )

        return (torch.tensor(quality), None)

    def short_name(self):
        return 'VMAFv1'        

    @staticmethod
    def name():
        return 'VMAF_v1'

    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 113

