# Video-Segmentation-for-Autonomous-Manipulation
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

The segmentation masks are intended as inputs to an imitation
learning policy for autonomous manipulation with the
third surgical arm. We present quantitative comparisons
of segmentation quality, model latency, and qualitative
outputs across different methods, highlighting our U-Net’s
balance between performance and efficiency. This work
contributes a deployable perception module tailored for
surgical robotics and paves the way toward real-time
learning-based automation in high-stakes environments.

Results:
Real Time Segmentation at 30 Hz Using UNet trained on SAM2 outputs:

https://github.com/user-attachments/assets/16ce64cb-83c8-4de6-8a5d-0fc8c2055da5

