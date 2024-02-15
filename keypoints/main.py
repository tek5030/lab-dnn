import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import cv2 as cv
import argparse


# SuperPoint+LightGlue

class SIFTMatcher:
    def __init__(self):
        self.sift = cv.SIFT_create()
        self.flann = cv.FlannBasedMatcher(dict(algorithm=0, trees=20), dict(checks=150))

    def __call__(self, img0, img1):
        # find the keypoints and descriptors with SIFT
        keypoint1, descriptors1 = self.sift.detectAndCompute(img0, None)
        keypoint2, descriptors2 = self.sift.detectAndCompute(img1, None)

        # finding nearest match with KNN algorithm
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)

        # Need to draw only good matches, so create a mask
        good_matches = [[0, 0] for i in range(len(matches))]

        # Good matches
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.5 * n.distance:
                good_matches[i] = [1, 0]

        return keypoint1, keypoint2, matches


class DNNMatcher:
    def __init__(self):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(DEVICE)  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().to(DEVICE)  # load the matcher

    def frame_to_torch(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype('float32') / 255.0
        frame = torch.from_numpy(frame).float()
        frame = frame[None, None].cuda()
        return frame

    def __call__(self, img0, img1):
        img0 = self.frame_to_torch(img0)
        img1 = self.frame_to_torch(img1)

        with torch.no_grad():
            feats0 = self.extractor(img0)
            feats1 = self.extractor(img1)

        with torch.no_grad():
            matches = self.matcher({
                'image0': feats0,
                'image1': feats1
            })
        matches = rbd(matches)
        matches = matches['matches']  # indices with shape (K,2)

        points0 = feats0['keypoints'][0][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][0][matches[..., 1]]
        cv_points0 = [cv.KeyPoint(p[0], p[1], 1) for p in points0.cpu().numpy()]
        cv_points1 = [cv.KeyPoint(p[0], p[1], 1) for p in points1.cpu().numpy()]
        cv_matches = [(cv.DMatch(i, i, 0), cv.DMatch(i, i, 0)) for i in range(len(points0))]
        return cv_points0, cv_points1, cv_matches


def main(args):
    sift_matcher = SIFTMatcher()
    dnn_matcher = DNNMatcher()
    matcher = sift_matcher if args.matcher == 'sift' else dnn_matcher

    cap = cv.VideoCapture(0)
    frame_prev = None
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frame_cur = frame.copy()
        if frame_prev is None:
            frame_prev = frame_cur
            continue
        keypoint0, keypoint1, matches = matcher(frame_prev, frame_cur)
        if len(matches) > 0:
            # Draw the matches using drawMatchesKnn()
            show_matched = cv.drawMatchesKnn(
                frame_prev,
                keypoint0,
                frame_cur,
                keypoint1,
                matches,
                outImg=None,
                matchColor=(0, 155, 0),
                singlePointColor=(0, 255, 255),
                flags=0)
            cv.imshow('Matches', show_matched)
        frame_prev = frame_cur


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--matcher', type=str, default='dnn', help='matcher to use')

    main(args=argparse.parse_args())
