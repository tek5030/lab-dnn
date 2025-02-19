import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import cv2 as cv

def scale_for_cv(image):
    image = image.copy()
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype('uint8')

class DepthDisplay:
    def __init__(self):
        self.current_depth = None

    def update_depth(self, depth):
        self.current_depth = depth

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.current_depth is not None:
                h, w = self.current_depth.shape
                if y < h and x < w:
                    depth_value = self.current_depth[y, x]
                    print(f"Depth at ({x}, {y}) is: {depth_value:.2f} meters")
                else:
                    print("Clicked position is out of bounds.")

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    # model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf").to(DEVICE)
    transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf").to(DEVICE)
    # transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    # model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf").to(DEVICE)

    depth_display = DepthDisplay()
    cv.namedWindow('Depth')
    cv.setMouseCallback('Depth', depth_display.on_mouse)

    cap = cv.VideoCapture(0)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        input_img = transform(images=cv.cvtColor(frame, cv.COLOR_BGR2RGB), return_tensors='pt')['pixel_values'].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_img)
            prediction = outputs.predicted_depth[0].cpu().numpy()
        depth_display.update_depth(prediction)

        show_prediction = scale_for_cv(prediction)
        cv.imshow('Input', frame)
        cv.imshow('Depth', show_prediction)
    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
