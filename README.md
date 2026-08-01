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
    - `scripts/generate_dataset_staircase2.sh` is a worked example for a second scene (`scenes/staircase2/`, from [Benedikt Bitterli's Rendering Resources](https://benedikt-bitterli.me/resources/)), with resolution/view-count/spp overridable via `WIDTH`/`HEIGHT`/`NUM_VIEWS`/`INPUT_SPP`/`AOV_SPP`/`REF_SPP` env vars — see [MANUAL_GUIDE.md §6](MANUAL_GUIDE.md#6-새-씬으로-데이터셋-생성하기-예-staircase2) for how to run it outside Claude Code (camera-matrix extraction, safety-mode docker invocation, output layout).

- Training the JSA and JSA+Conv models
    - `train.sh`
    - `train_conv.sh`
    - Each scene gets its own config module and training entry point, so training on a new scene never overwrites another scene's config/checkpoints. For `scenes/staircase2/`: `codes/config_staircase2.py` / `codes/config_cnn_staircase2.py` (parallel to `config.py`/`config_cnn.py`, only the dataset paths/task name differ), run via `scripts/train_staircase2.sh` / `scripts/train_conv_staircase2.sh`.

- Compiling trained `.pth` checkpoints with TensorRT

- Comparing JSA vs JSA+Conv on a test view (GT / noisy / JSA / JSA+Conv panel with PSNR/SSIM/inference-time)
    - `scripts/compare_conv.sh --view-index N` (classroom), `scripts/compare_conv_staircase2.sh --view-index N` (staircase2)
    - See [MANUAL_GUIDE.md §4](MANUAL_GUIDE.md#4-이미지-비교-gt--noisy--jsa--jsaconv) for options and output layout.

- Running the interactive web viewer for inference visualization, metric evaluation, and generation of new test data
    - [Further documentation](ViewerDocuments_forJSA/viewer.md)

- Running the end-to-end pipeline, from dataset generation to launching the web viewer
    - `run_all.sh`

- Manually reproducing the GTX 1650 (4GB) lightweight-port workflow (training, TensorRT benchmarking, image/metric comparison) step by step in a terminal
    - [Further documentation](MANUAL_GUIDE.md)