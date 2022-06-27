# My-graduation-design
For my graduation project, I've done a reserach on image reconstruction from optical speckle patterns. Both iterative phase retrieval algorithm and deep learning method for end-to-end mapping are employed.

**Abstract** 
>Imaging through scattering complex media is a pervasive problem encountered in many cases, such as deep tissue optical imaging. Light scattering scrambles the waves propagating inside the media, with the result of no image of the target being produced, but only seemingly random pattern known as speckle. For computational imaging way, major progresses have been made by using transmission matrix (TM) method to descramble the speckle patterns in order to ‘look through’ a scattering medium. Recently, deep learning (DL) techniques have been increasingly employed in this scenario. In this thesis, the research focus is to reconstruct target images from intensity-only measurements of speckle patterns. The first step is to obtain image dataset that contains target images and corresponding speckle patterns. Optical experiment was conducted where the MNIST handwritten digit images were displayed on a phase-only Spatial Light Modulator (SLM) to manipulate the laser incident onto a diffuser, with the resulting speckle patterns being measured by a camera. Apart from the MNIST dataset, a public dataset containing empirical TM of scattering medium was also downloaded. Based on the downloaded dataset, phase retrieval algorithm employing TM measurement operator was able to retrieve SLM patterns successfully. What’s more, a modified U-net model employing dense blocks has been utilized to efficiently reconstruct the MNIST images displayed on SLM from the recorded speckle intensity patterns. Reconstruction results on test set were compared qualitatively and quantitatively when the network was trained using MSE and NPCC loss function, respectively. Further recognition for speckle patterns was tried using ResNet18 via transfer learning, which could demonstrate that seeming random speckle patterns actually contain information about the input digit images.

**This repository contains my graduation thesis, PPT presentation for thesis defense, also some MATLAB and Python codes.**
1. Iterative_GS_algorithm.m: MATLAB demo for iterative Phase Retrieval from optical diffraction pattern, with reference to [toy phase retrieval alogrithm in MATLAB](https://github.com/necroen/toy_pr), which contains several variants: ER, HIO, DM, ASR, RAAR.
2. Using GS phase retrieval algorithm to recover object from speckle intensity measurement.
- GS_expTM.m: Both the TM (as the measurement matrix) and the speckle pattern are experimentally acquired, which are available from a public dataset. We refer to Ref[1] and its project home [Coherent Inverse Scattering via Transmission Matrices](http://compphotolab.northwestern.edu/project/transmissionmatrices/) where the dataset and some scripts can be downloaded.
- GS_simTM.m: The TM is not a experimentally acquired but instead a simulated one. It can run immedicately without any extra requirements. Also, this version should be easier to understand.

3. unet1.py & Rmain_GPU.py: Modified “U-Net” for image reconstruction from speckle pattern. Here, unet1.py is the DNN model established under Pytorch framework. Rmain_GPU.py descirbes the data loading, network training and validation process. 
4. generalization_test.py: For generalization test, you can download the pretrained weight parameters(about 205MB), the test set(about 50MB) and corresponding image label(about 800KB) all saved in one folder [test](https://pan.baidu.com/s/1AxmDbcCSw8dAojpH5Skigw). After downloading the files, put them under the root directory and run generalization_test.py.










**Reference**

[1] Metzler, C.A., M.K. Sharma, S. Nagesh, R.G. Baraniuk, O. Cossairt, and A. Veeraraghavan, "Coherent Inverse Scattering via Transmission Matrices: Efficient Phase Retrieval Algorithms and a Public Dataset," 2017 Ieee International Conference on Computational Photography (Iccp 2017), 2017: 51-66.
