# Real Time Video-Segmentation-for-Autonomous-Manipulation
We propose a real-time semantic segmentation framework
for robotic surgical scenes, enabling downstream
imitation learning for autonomous robotic assistance. Our
system is built on top of a custom-collected dataset from
the daVinci Surgical System performing object transfer
tasks. We leveraged the Segment Anything Model 2 (SAM2)
to obtain high-quality masks and trained a lightweight
U-Net architecture on them, achieving near-equivalent
segmentation performance with 30Hz inference speed,
suitable for closed-loop robotic control.

Model Pipeline
**Figure 1: Pseudo-Ground-Truth Generation using SAM2**  
<img width="827" alt="SAM2" src="https://github.com/user-attachments/assets/f6ae3529-fdb3-4ce7-b23d-6e821b29eeed" />

**Figure 2: Fast Segmentation with U-Net**  
<img width="845" alt="U-Net" src="https://github.com/user-attachments/assets/b47083f1-2ad0-41f3-a80f-da43c703dd71" />

**Figure 3: Real-Time Robot Perception Action Loop**  
<img width="834" alt="Robot Loop" src="https://github.com/user-attachments/assets/f33ff8d1-f7e7-464f-80da-5b9c29d8f843" />

Results:
Real Time Segmentation at 30 Hz Using UNet trained on SAM2 outputs:

https://github.com/user-attachments/assets/16ce64cb-83c8-4de6-8a5d-0fc8c2055da5

