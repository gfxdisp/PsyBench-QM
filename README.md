# PyST - PyTorch Synthetic tests
The code is used to evaluate the quality metrics' ability to align with the psychophysical data and models of low-level human vision.
The code can be only run using **CPU** or **CUDA**.

## Installation quickstart
1. Start by creating a new environment for PyST and activate it:
```commandline
python -m venv .pyst
source .pyst/bin/activate
```
2. Install PyTorch by following [these instructions](https://pytorch.org/get-started/locally/).

3. Install dependencies:
```commandline
pip install -r requirements.txt
```

## How to add a metric to be evaluated with PyST
We provide some metrics in the **./metrics** folder. 

To include a new metric, you need to create a class for the metric that inherits **VQM**. Please find an example of TOPIQ metric below.
```python
import torch
from pyiqa import create_metric
from pyst.VQM import VQM

class topiq_metric(VQM):

    def __init__(self, device=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        """ 
        We are using ColorVideoVDP as the backend to process our data.
        It offers multiple colorspaces:
        * display_encoded_dmax: sRGB encoding
        * RGB709: Linear RGB709
        * RGB2020: Linear RGB2020
        * XYZ
        * LMS2006
        * DKLd65
        Please select the colorspace that the metric requires. For most metrics it is: dispaly_encoded_dmax.
        """
        self.color_space = 'display_encoded_dmax'
        # Define your metric
        self.metric = create_metric('topiq_fr', device=self.device)

    def predict_video_source(self, vid_source, frame_padding="replicate"):
        # Our code would provide the metric with a video source, you can use the following code to extract the frames and run your metric on it. 
        _, _, N_frames = vid_source.get_video_size()
        score = 0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2).clamp(min=0, max=1)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2).clamp(min=0, max=1)
            score += self.metric(T, R) / N_frames
        return score, None

    # Define metric name
    def short_name(self):
        return "TOPIQ"
    @staticmethod
    def name():
        return "TOPIQ"
    # Define if lower predictions mean higher quality
    @staticmethod
    def is_lower_better():
        return False
    # Define the range of the metric predictions
    @staticmethod
    def predictions_range():
        return 0, 1
```
Insure to add your new metric class in the __init__ file. Here is an example:
```python
from metrics.topiq_metric import topiq_metric
```
If your metric requires a YUV file, we also offer function to convert the video source to a YUV file. 
```python
from pyst.yuv_utils import YUVWriter
import tempfile
from subprocess import Popen, PIPE
import os 

def predict_video_source(self, vid_source, frame_padding="replicate"):

    h, w, N_frames = vid_source.get_video_size()

    """ Define your YUV file metadata """
    vprops = dict()
    vprops["width"]=w
    vprops["height"]=h
    vprops["fps"] = vid_source.get_frames_per_second()
    vprops["chroma_ss"] = '444'
    vprops["bit_depth"] = 16  # We use 16 bits, to offer the best precision possible, as we cannot offer float-points to the metric, as we are doing in other metrics!
    vprops["color_space"] = '709'
    
    name = 'metric_name'
    cache = 'cache_folder'
  
    with tempfile.TemporaryDirectory(prefix=name, dir=cache) as temp_dir:
        with YUVWriter( os.path.join( temp_dir, "test" ), vprops ) as test_vw, YUVWriter( os.path.join( temp_dir, "ref" ), vprops ) as ref_vw:

            for ff in range(N_frames):
                T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace).squeeze().permute(1,2,0).cpu().numpy()
                R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace).squeeze().permute(1,2,0).cpu().numpy()

                # Save the output as yuv file
                test_vw.append_frame_rgb(T)
                ref_vw.append_frame_rgb(R)
            
            """ Here we can run whatever metric using the YUV files: test_vw and ref_vw and get the quality score"""
            quality_score = ...

    return (quality_score, None)
```

## How to get the metric(s) results on the synthetic tests
We provide three main python files: 
```commandline
python run_synthetic_test.py
python run_synthetic_test_alignment_score.py
python get_alignment_scores.py
```
The first is used to generate the data for the figures, while the subsequent codes are used to calculate the error in each test. 

In the first and second codes you will need to specify the metrics (as you've defined previously), and specify the tests to run. Here is an example:
```python
from pyst.ContrastDetection import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from metrics import *

tests = [GaborPatchSpatialFreqAchTest, GaborPatchLuminanceAchTest, GaborPatchSizeAchTest, ContrastMaskingCoherentAchTest, ContrastMaskingIncoherentAchTest]
metrics = [topiq_metric]
```
You can specify as many tests and metrics to run.
```python
""" Achromatic tests"""
tests = [GaborPatchSpatialFreqAchTest, GaborPatchLuminanceAchTest, GaborPatchSizeAchTest, ContrastMaskingCoherentAchTest, ContrastMaskingIncoherentAchTest, ContrastConstancySpatialFreqAchTest]

""" Color tests"""
tests = [GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest, ContrastConstancyColorTest]

""" Temporal tests"""
tests = [GaborPatchSpatialFreqAchTransientTest, DiskTemporalFreqAchTest]

""" ContrastConstancySpatialFreqAchTest and ContrastConstancyColorTest cannot be run on run_synthetic_test_alignment_score.py """
```


To generate the webpage with all the results. Please run the following code:
```commandline
python generate_synthetic_test_webpage_plotly.py
```
In this code, as previously mentioned you will also need to specify the metrics and the tests before running the code. 

We provide an example of the metrics in our code from previous results we got that you can use to directly generate a webpage.
The results of the webpage can be found in **./synthetic_test_webpage_plotly/index.html**, which is provided with the supplementary (but including more metrics).