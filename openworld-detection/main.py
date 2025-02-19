import threading

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2 as cv

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
            prompt = input("Enter bounding box search prompt: ").strip()
            if prompt:
                print(f"Received prompt: {prompt}")
                self.last_prompt = prompt
            else:
                print("Empty prompt entered. Please try again.")

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
    prompt_listener = PromptListener()
    prompt_listener.start()
    cap = cv.VideoCapture(0)
    while cv.waitKey(10) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        inputs = processor(images=cv.cvtColor(frame, cv.COLOR_BGR2RGB), text=prompt_listener.last_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.3,
            target_sizes=[frame.shape[:2]]
        )[0]
        for score, box, label in zip(results['scores'], results['boxes'], results['labels']):
            x_min, y_min, x_max, y_max = map(int, box)
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            text = f"{label}: {score:.2f}"
            (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(frame, (x_min, y_min - text_height - 4), (x_min + text_width, y_min), (0, 255, 0), -1)
            cv.putText(frame, text, (x_min, y_min - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.imshow('Input', frame)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()