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

## ReadMe For Web-veiwer based interactive JSA inference test
Contains several scripts for
- Render an input-reference pair
    - scripts/generate_dataset.sh
    - you can use other scene file, with extract camera parameters from the XML parts for replacing cam pose for the scene
- Training JSA and JSA + Conv model
    - train.sh and train_conv.sh
- Compile trained pth via tensorRT
- Interactive viewer based inference visualization and measuring statistics including renderer to generate new test data
- and End to end running from generating dataset to open web viewer
    - run_all.sh