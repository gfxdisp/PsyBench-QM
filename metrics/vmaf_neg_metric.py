import json
import numpy as np
import os, os.path as osp
from time import time
from tqdm import trange
import torch
import tempfile
from subprocess import Popen, PIPE


from pycvvdp.vq_metric import vq_metric
from pyst.yuv_utils import YUVWriter

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
class vmaf_neg_met(vq_metric):
    def __init__(self, cache_ref_loc='./cache', device=None):
 
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.cache_ref_loc = cache_ref_loc

        self.color_space = 'display_encoded_dmax'

 
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

            out_fname = os.path.join( temp_dir, "vmaf.json" )

            with YUVWriter( os.path.join( temp_dir, "test" ), vprops ) as test_vw, YUVWriter( os.path.join( temp_dir, "ref" ), vprops ) as ref_vw:

                #for ff in trange(N_frames, leave=False):
                for ff in range(N_frames):
                    T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.color_space).squeeze().permute(1,2,0).cpu().numpy()
                    R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.color_space).squeeze().permute(1,2,0).cpu().numpy()

                    # Save the output as yuv file
                    test_vw.append_frame_rgb(T)
                    ref_vw.append_frame_rgb(R)

                vmaf_cmd = f'vmaf --pixel_format 444 --width {w} --height {h} --bitdepth 10 --distorted {test_vw.fname} --reference {ref_vw.fname} --json -o {out_fname} --model version=vmaf_v0.6.1neg --threads 32 --quiet'

                os.system(vmaf_cmd)
                with open(out_fname) as f:
                    results = json.load(f)
                    quality = results['pooled_metrics']['vmaf']['mean']

        return (torch.tensor(quality), None)

    def short_name(self):
        return 'VMAF_v0.6.1neg'

    @staticmethod
    def name():
        return 'VMAF_v0.6.1neg'

