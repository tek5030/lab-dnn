import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import cv2 as cv

def scale_for_cv(image):
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype('uint8')

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    # model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf").to(DEVICE)
    transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf").to(DEVICE)
    # transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    # model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf").to(DEVICE)

    cap = cv.VideoCapture(0)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        #frame = frame / 255.0
        # frame = transform({"image": frame})["image"]
        input_img = transform(images=cv.cvtColor(frame, cv.COLOR_BGR2RGB), return_tensors='pt')['pixel_values'].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_img)
            prediction = outputs.predicted_depth[0]
        print(prediction.min().item())
        show_prediction = scale_for_cv(prediction.cpu().numpy())
        cv.imshow('Input', frame)
        cv.imshow('Depth', show_prediction)




if __name__ == '__main__':
    main_v2()
