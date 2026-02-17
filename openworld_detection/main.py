import threading
import cv2 as cv
import torch

try:
    from ultralytics import YOLO, YOLOE
except ImportError:
    raise ImportError("Please install ultralytics with `pip install ultralytics`")

class PromptListener:
    def __init__(self):
        self.last_prompt = ""
        # Create a daemon thread that will run the prompt loop.
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        # Start the thread.
        self.thread.start()

    def run(self):
        # Continuously listen for user input from the command line.
        while True:
            prompt = input("Enter bounding box search prompt (separate multiple classes with ','): ").strip()
            if prompt:
                print(f"Received prompt: {prompt}")
                self.last_prompt = prompt
            else:
                print("Empty prompt entered. Please try again.")

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Load the requested YOLOE-26N model
    # Assuming 'yoloe-26n.pt' is the correct model name/path expected by ultralytics
    # If this is a custom model, ensure the .pt file is in the path.
    # If it is a generic YOLO-World model, it might be named differently (e.g. yolov8s-world.pt)
    # but we follow the user's request.
    try:
        model = YOLOE("yoloe-26l-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes

    except Exception as e:
        print(f"Error loading model 'yoloe-26l-seg.pt': {e}")
        print("Falling back to 'yolov8n-world.pt' for demonstration purposes if YOLOE-26N is unavailable.")
        model = YOLO("yolov8n-world.pt")

    prompt_listener = PromptListener()
    prompt_listener.start()
    
    cap = cv.VideoCapture(0)
    
    current_prompt = None
    
    while cv.waitKey(10) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
            
        # Update model classes if prompt has changed
        if prompt_listener.last_prompt != current_prompt:
            current_prompt = prompt_listener.last_prompt
            if current_prompt:
                # Split prompts by ',' to support multiple classes (GroundingDINO style)
                classes = [c.strip() for c in current_prompt.split(',') if c.strip()]
                print("Detected classes:", end=" ")
                for c in classes:
                    print(f"{c}", end=" ")
                print()
                if classes:
                    # Set the classes for Open World detection
                    try:
                        model.set_classes(classes)
                    except AttributeError:
                        # Fallback for models that might not support set_classes directly in the same way
                        # But YOLO-World models in ultralytics support this.
                        pass
                else:
                     # If parsed classes are empty, maybe reset?
                     pass

        # Perform inference
        # Only predict if we have set classes or if the model has default classes
        # For Open Vocabulary, we really need classes set.
        if current_prompt:
            results = model.predict(frame, conf=0.2, iou=0.5, verbose=False, device=DEVICE)
            
            # Draw results
            for result in results:
                for box in result.boxes:
                    # Get box coordinates
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get score and label
                    score = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    
                    # Draw rectangle
                    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
                    # Draw label and score
                    text = f"{label}: {score:.2f}"
                    (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 0.5, 1)
                    cv.rectangle(frame, (x_min, y_min - text_height - 4), (x_min + text_width, y_min), (0, 255, 0), -1)
                    cv.putText(frame, text, (x_min, y_min - 2), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        
        cv.imshow('Input', frame)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
