# Join Self-Attention Denoising and Real-time project for ADD repository

## [Joint Self-Attention for Denoising Monte Carlo Rendering, original paper project page](https://cglab.gist.ac.kr/visualcomputer24jsa/)

[Geunwoo Oh](https://greeny53.notion.site/Geunwoo-Oh-f110abf14321482d9dbc435982faa5ef), [Bochang Moon](https://cglab.gist.ac.kr/people/bochang.html)
 
![Teaser](teaser.png)

An official source code of The Visual Computer paper, "Joint Self-Attention for Denoising Monte Carlo Rendering".

For more information, please refer to our project page or other resources below:
* [project page](https://cglab.gist.ac.kr/visualcomputer24jsa/)
* [paper](https://link.springer.com/article/10.1007/s00371-024-03446-8)


## [Technical report for proposed paper(KIMST 2026, Juhwan Bae et al.)](TechnicalReport_Paper/docs.md)

Detailed description for the background and methods for the paper, written by author

## README for Web-Based Interactive JSA Inference

This directory contains scripts for:

- Rendering an input-reference pair
    - `scripts/generate_dataset.sh`
    - To use a different scene, extract the camera parameters from the scene XML and replace the camera pose.

- Training the JSA and JSA+Conv models
    - `train.sh`
    - `train_conv.sh`

- Compiling trained `.pth` checkpoints with TensorRT

- Running the interactive web viewer for inference visualization, metric evaluation, and generation of new test data
    - [Further documentation](ViewerDocuments_forJSA/viewer.md)

- Running the end-to-end pipeline, from dataset generation to launching the web viewer
    - `run_all.sh`