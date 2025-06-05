import cv2
import torch
import numpy as np
from PIL import Image
from utils import UNet, SegmentationTransform
import time


class LiveSegmenter:
    def __init__(self, checkpoint_path, device=None, video_source=0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)
        self.transform = SegmentationTransform()
        self.video_source = video_source
        self.cap = None

    def _load_model(self, checkpoint_path):
        model = UNet().to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        print(f"[INFO] Loaded model from {checkpoint_path}")
        return model

    def _preprocess(self, frame):
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        dummy_mask = Image.new("L", image_pil.size)
        image_tensor, _ = self.transform(image_pil, dummy_mask)
        return image_tensor.unsqueeze(0).to(self.device)

    def _postprocess(self, output, original_shape):
        pred_mask = (output > 0.5).float().squeeze().cpu().numpy()
        resized_mask = cv2.resize(pred_mask, original_shape, interpolation=cv2.INTER_NEAREST)
        mask_colored = cv2.cvtColor((resized_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return mask_colored

    def _segment_frame(self, frame):
        input_tensor = self._preprocess(frame)
        with torch.no_grad():
            output = self.model(input_tensor)
        return self._postprocess(output, (frame.shape[1], frame.shape[0]))

    def run(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video source.")

        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Define the output video path and writer
        output_path = self.video_source.replace(".mp4", "_segmented.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

        print("[INFO] Starting live segmentation. Press 'q' to quit.")
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Failed to read frame.")
                break

            t0 = time.time()
            mask = self._segment_frame(frame)
            t1 = time.time()
            combined = np.hstack((frame, mask))
            cv2.imshow("Raw (left) | Segmentation (right)", combined)
            print("Image Processing Time: {:.3f} | Frame Number: {}".format(t1-t0, frame_num))

            # Write the combined frame to the output video
            out.write(combined)
            frame_num += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        out.release()  # Release the video writer
        cv2.destroyAllWindows()
        print(f"[INFO] Stream ended. Segmented video saved to {output_path}")

if __name__ == "__main__":
    checkpoint_path ="/home/chetan/Desktop/Acads/CS231n/Project/Video-Segmentation-for-Autonomous-Manipulation/UNet/Saved Models/working.pth"
    video_source = "/home/chetan/Downloads/left.mp4"
    segmenter = LiveSegmenter(checkpoint_path, video_source=video_source)
    segmenter.run()
