# https://docs.opencv.org/4.9.0/d2/d58/tutorial_table_of_content_dnn.html
# https://github.com/opencv/opencv/blob/4.x/samples/dnn/edge_detection.py
# https://docs.opencv.org/4.9.0/dc/db1/tutorial_dnn_custom_layers.html

# https://docs.opencv.org/4.9.0/d5/de7/tutorial_dnn_googlenet.html

# https://docs.opencv.org/4.9.0/d0/dd4/tutorial_dnn_face.html

import cv2 as cv
import argparse

parser = argparse.ArgumentParser(
    description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--prototxt', help='Path to deploy.prototxt', required=True)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel', required=True)
parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
args = parser.parse_args()


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cv.dnn_registerLayer('Crop', CropLayer)

    # Load the model.
    net = cv.dnn.readNet(cv.samples.findFile(args.prototxt), cv.samples.findFile(args.caffemodel))

    kWinName = 'Holistically-Nested Edge Detection'
    cv.namedWindow('Input', cv.WINDOW_NORMAL)
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)

    cap = cv.VideoCapture(args.input if args.input else 0)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        cv.imshow('Input', frame)

        inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                                   mean=(104.00698793, 116.66876762, 122.67891434),
                                   swapRB=False, crop=False)
        net.setInput(inp)

        out = net.forward()
        out = out[0, 0]
        out = cv.resize(out, (frame.shape[1], frame.shape[0]))
        cv.imshow(kWinName, out)
