import cv2
import numpy as np
import yaml
import argparse
import os

class HSVMaskBuilder:
    def __init__(self, config_dict):
        self.video_path = config_dict["video_path"]
        self.hsv_yaml_path = config_dict["hsv_bounds_yaml_path"]

        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([179, 255, 255])

        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.first_frame = None
        self._load_first_frame()

        # Flow control: Load HSV if YAML exists; otherwise, go through tuning first
        if self.hsv_yaml_path and os.path.exists(self.hsv_yaml_path):
            self._load_hsv_bounds(self.hsv_yaml_path)
            self.apply_mask_to_video()
        else:
            self.create_sliders()
            self.show_mask()
            self.save_hsv_bounds(self.hsv_yaml_path)
            self.apply_mask_to_video()

    def _load_first_frame(self):
        ret, frame = self.capture.read()
        if ret:
            self.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("Unable to read first frame from video.")

    def _load_hsv_bounds(self, hsv_bounds_path):
        try:
            with open(hsv_bounds_path, "r") as file:
                hsv_bounds = yaml.safe_load(file)
                self.hsv_lower = np.array(hsv_bounds["hsv_lower"])
                self.hsv_upper = np.array(hsv_bounds["hsv_upper"])
                print(f"✅ Loaded HSV bounds from {hsv_bounds_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load HSV bounds: {e}")

    def _update_lower_h(self, value): self.hsv_lower[0] = value
    def _update_lower_s(self, value): self.hsv_lower[1] = value
    def _update_lower_v(self, value): self.hsv_lower[2] = value
    def _update_upper_h(self, value): self.hsv_upper[0] = value
    def _update_upper_s(self, value): self.hsv_upper[1] = value
    def _update_upper_v(self, value): self.hsv_upper[2] = value

    def create_sliders(self):
        cv2.namedWindow("HSV Sliders")
        cv2.createTrackbar("Lower H", "HSV Sliders", 0, 179, self._update_lower_h)
        cv2.createTrackbar("Lower S", "HSV Sliders", 0, 255, self._update_lower_s)
        cv2.createTrackbar("Lower V", "HSV Sliders", 0, 255, self._update_lower_v)
        cv2.createTrackbar("Upper H", "HSV Sliders", 179, 179, self._update_upper_h)
        cv2.createTrackbar("Upper S", "HSV Sliders", 255, 255, self._update_upper_s)
        cv2.createTrackbar("Upper V", "HSV Sliders", 255, 255, self._update_upper_v)

    def show_mask(self):
        while True:

            mask = cv2.inRange(self.first_frame, self.hsv_lower, self.hsv_upper)
            result = cv2.bitwise_and(self.first_frame, self.first_frame, mask=mask)
            # hsv_frame = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2HSV)
            gray_frame = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("HSV", hsv_frame)

            cv2.imshow("Gray", gray_frame)

            cv2.imshow("Mask", mask)
            cv2.imshow("Result", cv2.cvtColor(result, cv2.COLOR_HSV2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def apply_mask_to_video(self):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, self.hsv_lower, self.hsv_upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert mask to 3 channels and colorize (e.g., red overlay)
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_color[np.where((mask_color != [0,0,0]).all(axis=2))] = [0,0,255]  # Red overlay

            # Blend the original frame and the colored mask
            blended = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("HSV", hsv_frame)
            cv2.imshow("Result", result)
            cv2.imshow("Blended Mask", blended)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()

    def save_hsv_bounds(self, output_path):
        hsv_bounds = {
            "hsv_lower": self.hsv_lower.tolist(),
            "hsv_upper": self.hsv_upper.tolist()
        }
        with open(output_path, "w") as file:
            yaml.dump(hsv_bounds, file)
        print(f"HSV bounds saved to {output_path}")


# ---------------------- Usage ----------------------
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="HSV Mask Builder")
    # parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    # parser.add_argument("--hsv_yaml_path", type=str, required=True, help="Path to load/save HSV bounds YAML")

    # args = parser.parse_args()
    data_folder = "/Users/chetan/Desktop/Spring 2025/CS231n/Project/Video-Segmentation-for-Autonomous-Manipulation/Data/Demo1/"
    video_path = os.path.join(data_folder,"videos","left.mp4")
    hsv_bounds_yaml_path = os.path.join(data_folder,"videos","hsv_bounds_arm.yaml")
    config_dict = {
        "video_path": video_path,
        "hsv_bounds_yaml_path": hsv_bounds_yaml_path
    }

    HSVMaskBuilder(config_dict)
