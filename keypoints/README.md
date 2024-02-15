# Keypoint detection

Keypoints are used throughout the course and can be detected with a variety of methods, but the methods based
on deep learning have prooven to reliable, albeit not always the fastest.

![Keypoint detection](https://github.com/cvg/LightGlue/raw/main/assets/easy_hard.jpg)

Here we provide inference code for LightGlue, the full code can be found at [LightGlue](https://github.com/cvg/LightGlue).
We also provide a demo with sift features, for comparison.

## Prerequisites

```
pip install -r requirements.txt
```

## Usage

```
python main.py --matcher dnn
```