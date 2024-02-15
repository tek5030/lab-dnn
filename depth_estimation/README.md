# Monocular Depth Estimation

This is code for running a depth estimation network from [Depth-Anything](https://github.com/LiheYoung/Depth-Anything/tree/main). The network try to guess the depth of each pixel in an image, given a single image as input.

![Depth estimation](https://github.com/LiheYoung/Depth-Anything/raw/main/assets/teaser.png)

The depth is not given in meters, but in a relative scale. This means that the network will output a depth map where the values are proportional to the distance of the objects in the scene.

To estimate metric depth you can use the additional code provided in the [Depth-Anything/metric_depth](https://github.com/LiheYoung/Depth-Anything/tree/main).

## Prerequisites

```
pip install -r requirements.txt
```

## Usage

```
python main.py --encoder vits
```
