import os
import torch
import numpy as np
import cv2
import time
from sam2.build_sam import build_sam2_camera_predictor
import sys
sys.path.insert(1, "/home/vakula/vakula/cs231n/sam2_repo")

input_video_path = "./media/videos/left.mp4"
output_video_path = "./outputs/rt_seg_op1.mp4"
sam2_checkpoint = "./sam2_repo/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)

cap = cv2.VideoCapture(input_video_path)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read the first frame")

height, width = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
predictor.load_first_frame(first_frame_rgb)

first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

bbox_cv = cv2.selectROI("Select object to segment", first_frame_bgr, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select object to segment")

x, y, w, h = bbox_cv
bbox = np.array([[x, y], [x + w, y + h]], dtype=np.float32)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
    for i in range(len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        all_mask = cv2.bitwise_or(all_mask, out_mask)

    all_mask_color = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(frame, 1.0, all_mask_color, 0.5, 0)

    out.write(blended)
    frame_idx += 1

cap.release()
out.release()
print(f"Segmented video saved to {output_video_path}")
