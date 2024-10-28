# PyST - PyTorch Synthetic tests
The code is used to evaluate the quality metrics' ability to mimic the mechanism of the human visual system.

## PyTorch quikstart
1. Start by creating a new environment for PyST and activate it:
```commandline
python -m venv .pyst
source .pyst/bin/activate
```
2. Install PyTorch by following [these instructions](https://pytorch.org/get-started/locally/).
3. Install ffmpeg
```commandline
pip install ffmpeg
pip install ffmpeg-python
```
4. Install ColorVideoVDP by following [these instructions](https://github.com/gfxdisp/ColorVideoVDP)
5. Obtain the PyST codebase, by cloning the repository:
```commandline
git clone https://github.com/gfxdisp/PyST.git
```
6. Install dependencies (I think that is the only one for now at least):
```commandline
pip install imutils
```
7. Finally, run PyST (for now this should be good)
8. If you want to run the metrics I provide, please install [PyIQA](https://github.com/chaofengc/IQA-PyTorch) and [PIQ](https://github.com/photosynthesis-team/piq) using the following commands:
```commandline
pip install pyiqa
pip install piq
```

## How to use PyST
We provide two main codes: 
```commandline
PYTHONPATH=pyst python run_synthetic_test.py
PYTHONPATH=pyst python generate_synthetic_results_webpage.py
```

The first code run the metrics you define on the tests you select.
The second code generate the results webpage. 