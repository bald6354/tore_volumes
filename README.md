# tore_volumes

Source code for generating and testing TORE volumes

Contains a MATLAB implementation for our [paper](https://arxiv.org/abs/2103.06108).  If you find this code useful in your research, please consider citing:

    @article{baldwin2021time,
    title={Time-Ordered Recent Event (TORE) Volumes for Event Cameras},
    author={Baldwin, R and Liu, Ruixu and Almatrafi, Mohammed and Asari, Vijayan and Hirakawa, Keigo},
    journal={arXiv preprint arXiv:2103.06108},
    year={2021}
    }
 
![Missing Image](https://github.com/bald6354/tore_volumes/blob/main/pics/pos_shapes_vol3_cropped_scaled-removebg-preview.jpg "TORE Volume")

This code was tested on an Ubuntu 18.04 system (i7-8700 CPU, 64GB RAM, and GeForce RTX 2080Ti GPU) running MATLAB 2019b/2020b. Matlab's image processing, computer vision, ROS, and Deep learning toolboxes are required. Be sure to add the code folder to your Matlab path before running.

Information on the 2D to 3D human pose network can be found here: https://github.com/lrxjason/Attention3DHumanPose

Information on the EDnCNN denoise network can be found here: https://github.com/bald6354/edncnn

## DVSNOISE20 Dataset (used for image reconstruction and denoising)
To download the dataset use: [dvsnoise20](https://sites.google.com/a/udayton.edu/issl/software/dataset).

## Reading AEDAT data into MATLAB (used for reading some datasets)
To read AEDAT (jAER) data into MATLAB use: [AedatTools](https://gitlab.com/inivation/AedatTools) or [AedatTools Alt](https://github.com/simbamford/AedatTools/).

## Reading AEDAT4 data into MATLAB (optional)
To read AEDAT4 (DV) data into MATLAB use: [aedat4tomat](https://github.com/bald6354/aedat4tomat).
