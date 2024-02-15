import argparse

import torch

from depth_anything.dpt import DepthAnything
import cv2 as cv

from depth_estimation.depth_anything.util.transform import get_transform


def scale_for_cv(image):
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype('uint8')

def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{args.encoder}14").to(DEVICE).eval()
    transform = get_transform()

    cap = cv.VideoCapture(0)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        frame = frame / 255.0
        frame = transform({"image": frame})["image"]
        frame = torch.from_numpy(frame).float().to(DEVICE)
        frame = frame[None, ...]
        with torch.no_grad():
            prediction = model(frame)
        prediction = prediction[0].cpu().numpy()
        prediction = scale_for_cv(prediction)
        cv.imshow('Input', cv.cvtColor(frame[0].cpu().numpy().transpose(1, 2, 0), cv.COLOR_RGB2BGR))
        cv.imshow('Depth', prediction)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    main(args=parser.parse_args())
