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
FUNQUE metric. 

Dependencies:

The repository from https://github.com/abhinaukumar/funque must be cloned into python_metrics

The following packages are also needed:

pip install imutils opencv-python scikit-image scikit-learn PyWavelets

We also need libsvm:

sudo apt install libsvm-dev

This metric will need to decode test/reference each video to yuv first. Make sure you have enough space in /tmp, or better create ramdisc and pass the path run run_metric.py:

python run_metric.py ... --metric-cache /media/ramdisc/

To create a ramdisc, execute as root:

path=/media/ramdisc
mkdir -p $path
chmod 777 $path
mount -t tmpfs -o size=16048M tmpfs $path

"""
class funque_met(vq_metric):
    def __init__(self, cache_ref_loc='./cache', device=None):
 
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        

        self.colorspace = 'display_encoded_dmax'

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

        vprops = dict()
        vprops["width"]=w
        vprops["height"]=h
        vprops["fps"] = vid_source.get_frames_per_second()
        vprops["chroma_ss"] = '444'

        if self.display_photometry == 'sRGB':
            vprops["bit_depth"] = 8
            vprops["color_space"] = '709'
            pix_fmt = 'yuv444p'
        else:
            vprops["bit_depth"] = 10
            vprops["color_space"] = '2020'
            pix_fmt = 'yuv444p10le'

        with tempfile.TemporaryDirectory(prefix=self.short_name(), dir=self.cache_ref_loc) as temp_dir:
            with YUVWriter( os.path.join( temp_dir, "test" ), vprops ) as test_vw, YUVWriter( os.path.join( temp_dir, "ref" ), vprops ) as ref_vw:

                #for ff in trange(N_frames, leave=False):
                for ff in range(N_frames):
                    T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace).squeeze().permute(1,2,0).cpu().numpy()
                    R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace).squeeze().permute(1,2,0).cpu().numpy()

                    # Save the output as yuv file
                    test_vw.append_frame_rgb(T)
                    ref_vw.append_frame_rgb(R)

                funque_path='metrics/funque'
                funque_cmd = f'python {funque_path}/run_funque.py ' \
                            f'{pix_fmt} {w} {h} {test_vw.fname} {ref_vw.fname} --model {funque_path}/model/funque_release.json --out-fmt json'

                with Popen(funque_cmd.split(), stdin=None, stdout=PIPE) as proc:
                    res_str = proc.stdout.read()

        res = json.loads(res_str)
        quality = float(res["aggregate"]["FUNQUE_score"])

        return (torch.tensor(quality), None)

    def short_name(self):
        return 'FUNQUE'

    @staticmethod
    def name():
        return 'FUNQUE'

    @staticmethod
    def is_lower_better():
        return True

    @staticmethod
    def predictions_range():
        return [100, 0]

