# My-graduation-design
For my graduation project, I've done a reserach on image reconstruction from optical speckle patterns. Both iterative phase retrieval algorithm and deep learning method for end-to-end mapping are employed.

Abstract 
>Imaging through scattering complex media is a pervasive problem encountered in many cases, such as deep tissue imaging. Light scattering scrambles the waves propagating inside the media, with the result of no image of the target being produced, but only seemingly random pattern known as speckle. There have been tremendous experimental efforts by exploiting digital holography, wavefront shaping techniques or memory effect to explore imaging through scattering media, while complex optical setups are usually required. In the computational imaging scenario, major progresses have been made by using transmission matrix (TM) method to descramble the transmitted images across scattering media.  In the recent years, Deep Learning techniques have been increasingly employed for computational imaging through scattering media. Through the data-driven learning approach, deep learning can build a computational architecture as a generic function approximation of light propagation process across complex media. This opens up the avenue to reconstruct the target objects from the recorded speckle patterns via learning to map the input/output wavefronts through scattering media.  In this thesis, the research focus is to reconstruct target images from intensity-only measurement of speckle patterns. The first step is obtaining image dataset that contains target images and corresponding speckle patterns. Optical experiment was conducted where the MNIST handwritten digit images were displayed on a phase-only Spatial Light Modulator (SLM) to manipulate the light incident onto a diffuser, with the resulting speckle patterns being measured by a camera. Apart from MNIST dataset, another public empirical dataset was also downloaded. Before the attempt to use deep learning for image reconstruction, the research explored phase retrieval in the scenario of imaging through scattering media. Based on the public dataset, iterative GS algorithm employing empirical TM measurement operator was able to retrieve SLM patterns successfully. Last but not least, a modified U-net model has been utilized to efficiently reconstruct the MNIST images displayed on SLM from the recorded intensity of speckle patterns, and further recognition for the speckle patterns via a ResNet18 model. This demonstrate that seeming random speckle patterns contain information about the input digit images, rendering it possible for efficient reconstruction and recognition.

**This repository contains my graduation thesis, PPT presentation for thesis defense, also some MATLAB and Python codes**
1. Iterative_GS_algorithm.m: MATLAB demo for iterative Phase Retrieval from optical diffraction pattern, with reference to [toy phase retrieval alogrithm in MATLAB include: ER, HIO, DM, ASR, RAAR r](https://github.com/necroen/toy_pr).
2. GS_TM.m: Phase retrieval from recorded speckle patterns using GS algorithm with transmission matrix as the measurement operator. We refer to Ref[1] and its project home [Coherent Inverse Scattering via Transmission Matrices](http://compphotolab.northwestern.edu/project/transmissionmatrices/) where the dataset and some scripts can be downloaded.
3. unet.py & Rmain_GPU.py: Modified “U-Net” for image reconstruction from speckle pattern
4. ResNet.py & Cmain_GPU: ResNet18 for MNIST classification and recognition of speckle pattern









**Reference**

[1] Metzler, C.A., M.K. Sharma, S. Nagesh, R.G. Baraniuk, O. Cossairt, and A. Veeraraghavan, "Coherent Inverse Scattering via Transmission Matrices: Efficient Phase Retrieval Algorithms and a Public Dataset," 2017 Ieee International Conference on Computational Photography (Iccp 2017), 2017: 51-66.
