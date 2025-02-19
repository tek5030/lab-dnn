# Grounding DINO: Open-World Text-to-Image Detection
Read more in the [GroundingDino repo](https://github.com/IDEA-Research/GroundingDINO/tree/main)

## Introduction

Open-world detection is a paradigm in computer vision where models are designed to detect and localize objects beyond a fixed set of predefined categories. Unlike traditional closed-set detectors that only recognize classes seen during training, open-world detectors must generalize to new, unseen objects—a critical capability for real-world applications where the range of potential objects is virtually unlimited.

By leveraging natural language prompts alongside visual data, Grounding DINO "grounds" free-form text in the image domain. This means you can specify objects of interest using descriptive language at inference time, enabling the system to detect objects that were never explicitly labeled during training.

<img src="https://github.com/IDEA-Research/GroundingDINO/blob/main/.asset/arch.png" alt="gd_gligen" width="100%">

## Key Concepts

- **Open-World Detection:** The ability to recognize and localize novel or unknown objects outside the training set, providing flexibility for dynamic environments.
- **Text-to-Image Detection:** Merging language and vision, this technique allows models to use text descriptions to guide object detection in images.
- **Grounding DINO Architecture:** Utilizes a transformer-based backbone along with text embeddings to align semantic information from textual prompts with visual features, effectively bridging the gap between language and vision.

## Running

Install additional dependencies with:

    pip install -r requirements.txt

Run main on webcam id=0

    python main.py

You can then type in things to search for in the images, where each thing must be lower-case and end with a ".". E.g.:

    Enter bounding box search prompt: person. cup. whitebord.
