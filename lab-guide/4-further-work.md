# Step 4: Browse and experiment with models from other model zoos
 
Now, we have become familiar with the dnn module of OpenCV, and you are ready for greater challenges. Look at other model zoos and see if you can incorporate some of the models into a program of your own. 

One suggestion is to feed the output of one network as input to a new network.
[face_recognition_sface] from the OpenCV model zoo is an example of this.

We have added a set of models that we found to be missing in the OpenCV model zoo: [Depth estimation](../depth_estimation/README.md), [Human Pose Estimation](../human_pose_estimation/README.md), [Open World detection](../openworld_detection/README.md) and [Keypoint detection](../keypoints/README.md).

These examples are already completed, but can be extended only limited to your imagination.

You can also dive into the [TEK5030 PyTorch-tutorial]!

Here are some suggestions:

- [dnn/samples]
- [tutorials]
- https://github.com/onnx/models
- https://huggingface.co/docs/timm/quickstart

That's it for today! Good luck, and have a nice weekend : )



[face_recognition_sface]: ../opencv_zoo/models/face_recognition_sface/README.md

[dnn/samples]: https://github.com/opencv/opencv/tree/4.x/samples/dnn
[tutorials]: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html

[TEK5030 PyTorch-tutorial]: https://github.com/sigmunjr/TEK5030_deep_learning_torch
