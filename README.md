# PsyBench-QM - Psychophysical Benchmark for Quality Metrics
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
We provide some metrics in the `./metrics` folder. 

To include a new metric, you need to create a class for the metric that inherits class `VQM`. Please find an example for the TOPIQ metric below.
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
Your new metric is picked up automatically - there's no registration step. Any class defined anywhere under `./metrics` (in a subfolder if you like) that inherits from `VQM` (or `vq_metric` directly) is discovered by name (the string returned by `name()`) and immediately available to `run_and_plot.py` and `generate_webpage.py`. If your metric file fails to import (e.g. a missing optional dependency), only that one metric is skipped - a warning is logged and every other metric still works.

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

## How to compute metric(s) results on the tests

Tests and metrics are both auto-discovered and selected by name on the command line - no Python editing needed. To see what's available:
```commandline
python run_and_plot.py --list-tests
python run_and_plot.py --list-metrics
```

`run_and_plot.py` runs one or more metrics on one or more tests, and for each test/metric pair saves both a scores CSV and a plot into an output directory (created automatically if it doesn't exist):
```commandline
python run_and_plot.py \
    --tests GaborPatchSpatialFreqAchTest GaborPatchLuminanceAchTest GaborPatchSizeAchTest ContrastMaskingCoherentAchTest ContrastMaskingIncoherentAchTest \
    --metrics topiq_metric \
    --output-dir results/achromatic_tests
```

For larger or repeatable runs, put the same options in a JSON config file instead:
```json
{
  "tests": ["GaborPatchSpatialFreqRGTest", "GaborPatchSpatialFreqYVTest", "ContrastConstancyColorTest"],
  "metrics": ["topiq_metric"],
  "output_dir": "results/color_tests"
}
```
```commandline
python run_and_plot.py --config my_run.json
```

Pass `--alignment-score` to run tests in alignment-score mode (near-threshold condition sampling). Not every test supports this - `ContrastConstancySpatialFreqAchTest` and `ContrastConstancyColorTest` don't - unsupported tests are skipped with a clear error message rather than failing the whole run.

`run_and_plot.py` also computes each test's alignment score automatically - there's no separate scoring script to run. Which run produces a score depends on the test:
- Most tests (Gabor Patch, Contrast Masking, Disk) score by Spearman correlation between the psychophysical test's condition axis and the metric's predictions. This is only meaningful for the near-threshold condition sampling used in alignment-score mode, so it's computed **only when run with `--alignment-score`** - a normal run of these tests produces no alignment score.
- The Contrast Matching tests are the opposite: they don't support `--alignment-score` at all, so their **normal run** is what produces their (RMSE/error-based) alignment score.

Scores are shown in the plot title where available, and merged into `alignment_scores/<test>.csv`, accumulating across separate runs rather than being overwritten. See `compute_alignment_score()` in `pyst/synthetic_test.py` (default) and `pyst/ContrastMatching.py` (overrides) if you're adding a new test and want to define its own alignment score.

## How to generate the results webpage

To generate an interactive (Plotly) HTML report from results produced by `run_and_plot.py`:
```commandline
python generate_webpage.py \
    --tests GaborPatchSpatialFreqAchTest GaborPatchLuminanceAchTest GaborPatchSizeAchTest \
    --metrics topiq_metric \
    --input-dir results/achromatic_tests \
    --output-dir synthetic_test_webpage_plotly
```
As with `run_and_plot.py`, use `--list-tests` / `--list-metrics` to see the available names.

The repository includes the computed results for multiple metrics in `results_alignment_score` that can be used to generate an HTML report.

The results of the webpage can be found in `./synthetic_test_webpage_plotly/index.html`, which is provided with the supplementary (but including more metrics).
