# IVC_MDE
Drone footage monocular estimation for BU Spring 2023 Image and Video Computing course. 

AICrowd Competition
https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/mono-depth-perception

AICrowd Repo:
https://gitlab.aicrowd.com/aicrowd/challenges/suadd-2023/suadd23-monodepth-amazon/-/tree/main


# **[Scene Understanding for Autonomous Drone Delivery SUADD'23 - Mono Depth Perception](https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/mono-depth-perception)** - Starter Kit
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the SUADD 2023 - Mono Depth Perception **Starter kit**! It contains:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

Quick Links:

* [SUADD 2023 Mono Depth Perception - Competition Page](https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/mono-depth-perception)
* [Discussion Forum](https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/discussion)
* [SUADD 2023 Challenge Overview](https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23)


# Table of Contents
1. [About the Scene Understanding for Autonomous Drone Delivery Challenge](#about-the-scene-understanding-for-autonomous-drone-delivery-challenge)
2. [Evaluation](#evaluation)
3. [Baselines](#baselines) 
4. [How to test and debug locally](#how-to-test-and-debug-locally)
5. [How to submit](#how-to-submit)
6. [Dataset](#dataset)
7. [Setting up your codebase](#setting-up-your-codebase)
8. [FAQs](#faqs)

# About the Scene Understanding for Autonomous Drone Delivery Challenge

Unmanned Aircraft Systems (UAS) have various applications, such as environmental  studies, emergency responses or package delivery. The safe operation of fully autonomous  UAS requires robust perception systems. 

For this challenge, we will focus on images of a single downward camera to estimate the scene's depth and perform semantic segmentation. The results of these two tasks can help the development of safe and reliable autonomous control systems for aircraft. 

This challenge includes the release of a new dataset of drone images that will benchmark semantic segmentation and mono-depth perception. The images in this dataset comprise realistic backyard scenarios of variable content and have been taken on various Above Ground Level (AGL) ranges.

**This challenge aims to foster the development of fully autonomous Unmanned  Aircraft Systems (UAS).** 

To achieve this, it needs to overcome a multitude of challenges. To leverage fully  autonomous drone navigation, the device needs to understand both objects in a scene and the scale and distance to them. 

**This project's two key computer vision components are semantic segmentation and depth perception.**

With this challenge, we aim to inspire the Computer Vision community to develop new insights and advance state-of-the-art in perception tasks involving drone images.


## About the Mono Depth Estimation Task

Depth estimation measures the distance between the camera and the objects in the scene.  It is an important perception task for an autonomous aerial drone. Using two stereo cameras makes this task solvable with stereo vision methods. This challenge aims to  create a model that can use the information of a single camera to predict the depth of every pixel. 

The output of this task must be an image of equal size to the input image, in which every pixel contains a depth value.

# Evaluation

Models submitted to the Depth Estimation task will be evaluated accoring to the Scale invariant logarithmic error (SILog) and Abs Rel score. The submission should generate outputs that are valid depth values, non positive values or invalid values will result in a failed submission.

The exact code used for the calculation of SILog can be found in the `si_log` function in [`local_evaluation.py`](https://gitlab.aicrowd.com/aicrowd/challenges/suadd-2023/suadd-2023-depth-perception-starter-kit/-/blob/master/local_evaluation.py)

## Monodepth alignment with ground truth

Since monodepth outputs are generally relative depth maps, the evaluator performs an alignment step between prediction and ground-truth. Least squares was used for this alignment step (for further reference, please check [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341v3)).

# Baselines

You can find training code for the baseline [here](https://gitlab.aicrowd.com/aicrowd/challenges/suadd-2023/suadd23-monodepth-amazon). It uses the open source library [MiDaS](https://github.com/isl-org/MiDaS).

The `midas-baseline` branch in this repository includes inference code to run the trained model, which can be found at `my_submission/midas_predictor.py`.

# How to Test and Debug Locally

The best way to test your models is to run your submission locally.

You can do this by simply running  `python local_evaluation.py`. **Note that your local setup and the server evalution runtime may vary.** Make sure you mention setup your runtime according to the section: [How do I specify my dependencies?](#how-do-i-specify-my-dependencies)

# How to Submit

You can use the submission script `source submit.sh <submission name>`

More information on submissions can be found in [SUBMISSION.md](/docs/submission.md).

#### A high level description of the Challenge Procedure:
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23).
2. **Clone** this repo and start developing your solution.
3. **Train** your models on the SUADD dataset, and ensure local_evaluation.py works.
4. **Submit** your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com)
for evaluation (full instructions below). The automated evaluation setup
will evaluate the submissions against the test data to compute and report the metrics on the leaderboard
of the competition.


# Dataset

Download the public dataset for this Task using the link below, you'll need to accept the rules of the competition to access the data.

https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/mono-depth-perception/dataset_files


# Setting Up Your Codebase

AIcrowd provides great flexibility in the details of your submission!  
Find the answers to FAQs about submission structure below, followed by 
the guide for setting up this starter kit and linking it to the AIcrowd 
GitLab.

## FAQs

* How do I submit a model?
  * More information on submissions can be found at our [submission.md](/docs/submission.md). In short, you should push you code to the AIcrowd's gitlab with a specific git tag and the evaluation will be triggered automatically.

### How do I specify my dependencies?

We accept submissions with custom runtimes, so you can choose your 
favorite! The configuration files typically include `requirements.txt` 
(pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about this in [runtime.md](/docs/runtime.md).

### What should my code structure look like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:


```
.
├── aicrowd.json                # Add any descriptions about your model and gpu flag
├── apt.txt                     # Linux packages to be installed inside docker image
├── requirements.txt            # Python packages to be installed
├── local_evaluation.py         # Use this to check your model evaluation flow locally
└── my_submission               # Place your models and related code here
    ├── <Your model files>      # Add any models here for easy organization
    ├── aicrowd_wrapper.py      # Keep this file unchanged
    └── user_config.py          # IMPORTANT: Add your model name here
```

### How can I get going with an existing baseline?

See [baselines section](#baselines)

### How can I get going with a completely new model?

Train your model as you like, and when you’re ready to submit, implement the inference class and import it to `my_submission/user_config.py`. Refer to [`my_submission/README.md`](my_submission/README.md) for a detailed explanation.

Once you are ready, test your implementation `python local_evaluation.py`

### How do I actually make a submission?

The submission is made by adding everything including the model to git,
tagging the submission with a git tag that starts with `submission-`, and 
pushing to AIcrowd's GitLab. The rest is done for you!

For large model weight files, you'll need to use `git-lfs`

More details are available at [docs/submission.md](/docs/submission.md).

### How to use GPU

To use GPU in your submissions, set the gpu flag in `aicrowd.json`. 

```
    "gpu": true,
```

### Are there any hardware or time constraints?

Your submission will need to complete predictions on each **image** within **10 seconds**. Make sure you take advantage 
of all the cores by parallelizing your code if needed. Incomplete submissions will not be scored.

The machine where the submission will run will have following specifications:
* 4 vCPUs
* 16GB RAM
* (Optional) 1 NVIDIA T4 GPU with 16 GB VRAM - This needs setting `"gpu": true` in `aicrowd.json`

# 📎 Important links
- 💪 Challenge Page: https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/mono-depth-perception
- 🗣️ Discussion Forum: https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/mono-depth-perception/leaderboards
