from ultralytics.utils.plotting import Annotator
import cv2 as cv


def main():
    from ultralytics import YOLO

    model = YOLO('yolov8n-pose.pt')  # load an official model

    # Predict with the model
    cap = cv.VideoCapture(0)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        results = model(frame)[0]  # predict on an image

        show_img = results.orig_img if len(results) == 0 else None
        for r in results:
            show_img = r.plot(img=show_img)
        cv.imshow('pose', show_img)



if __name__ == '__main__':
    main()