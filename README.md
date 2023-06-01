# Human pose dataset anything (segmentation pseudo GT)


We have created this repository to share code using SAM (Segmentation Anything) for generating pseudo ground truth (GT) for individuals using human pose datasets.

## Tasks to be done
* Make CrowdPose & Human36M pseudo GT 
* Make Visualization Code
* Upload Pseudo GT annotation json file

## Contribution

* ChangHee Yang

## Installation

You can follow the instructions provided at [This](https://github.com/facebookresearch/segment-anything) to proceed with the installation.

Requires python>=3.8
```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Download Human Pose dataset
Pseudo ground truth (GT) for Human36M, MuCo, 3DPW, and MPII datasets were created using dataset annotations provided by Mr. Moon Kyung-sik. You can download them through the following link. 

[I2L-MeshNet](https://github.com/YangChangHee/I2L-MeshNet_RELEASE)

OCHuman, and MSCOCO datasets were obtained from their respective official websites. The links for downloading these datasets are as follows.

[MSCOCO](https://cocodataset.org/#home)

[OCHuman](https://github.com/liruilong940607/OCHumanApi)

LSP dataset was obtained from BEV, and the link for downloading the dataset is as follows.

[BEV-ROMP github](https://github.com/YangChangHee/ROMP)

# Visualization

## MPII

<p float="left">
  <img src="assets/MPII_input.jpg?raw=true" width="30.00%" />
  <img src="assets/MPII_2d_input.jpg?raw=true" width="30.00%" /> 
  <img src="assets/MPII_mask.jpg?raw=true" width="30.00%" />
</p>

<p float="left">
  <img src="assets/MPII_input1.jpg?raw=true" width="30.00%" />
  <img src="assets/MPII_2d_input1.jpg?raw=true" width="30.00%" /> 
  <img src="assets/MPII_mask1.jpg?raw=true" width="30.00%" />
</p>

## MSCOCO

<p float="left">
  <img src="assets/MSCOCO_input.jpg?raw=true" width="30.00%" />
  <img src="assets/MSCOCO_2d_input.jpg?raw=true" width="30.00%" /> 
  <img src="assets/MSCOCO_mask.jpg?raw=true" width="30.00%" />
</p>

<p float="left">
  <img src="assets/MSCOCO_input1.jpg?raw=true" width="30.00%" />
  <img src="assets/MSCOCO_2d_input1.jpg?raw=true" width="30.00%" /> 
  <img src="assets/MSCOCO_mask1.jpg?raw=true" width="30.00%" />
</p>

## LSP

<p float="left">
  <img src="assets/LSP_input.jpg?raw=true" width="30.00%" />
  <img src="assets/LSP_2d_input.jpg?raw=true" width="30.00%" /> 
  <img src="assets/LSP_mask.jpg?raw=true" width="30.00%" />
</p>


<p float="left">
  <img src="assets/LSP_input1.jpg?raw=true" width="30.00%" />
  <img src="assets/LSP_2d_input1.jpg?raw=true" width="30.00%" /> 
  <img src="assets/LSP_mask1.jpg?raw=true" width="30.00%" />
</p>

## OCHuman


<p float="left">
  <img src="assets/OC_input.jpg?raw=true" width="30.00%" />
  <img src="assets/OC_2d_input.jpg?raw=true" width="30.00%" /> 
  <img src="assets/OC_mask.jpg?raw=true" width="30.00%" />
</p>


<p float="left">
  <img src="assets/OC_input1.jpg?raw=true" width="30.00%" />
  <img src="assets/OC_2d_input1.jpg?raw=true" width="30.00%" /> 
  <img src="assets/OC_mask1.jpg?raw=true" width="30.00%" />
</p>

## MuCo


<p float="left">
  <img src="assets/MuCo_input.jpg?raw=true" width="30.00%" />
  <img src="assets/MuCo_2d_input.jpg?raw=true" width="30.00%" /> 
  <img src="assets/MuCo_mask.jpg?raw=true" width="30.00%" />
</p>


<p float="left">
  <img src="assets/MuCo_input1.jpg?raw=true" width="30.00%" />
  <img src="assets/MuCo_2d_input1.jpg?raw=true" width="30.00%" /> 
  <img src="assets/MuCo_mask1.jpg?raw=true" width="30.00%" />
</p>

## 3DPW


<p float="left">
  <img src="assets/3dpw_input.jpg?raw=true" width="30.00%" />
  <img src="assets/3dpw_2d_input.jpg?raw=true" width="30.00%" /> 
  <img src="assets/3dpw_mask.jpg?raw=true" width="30.00%" />
</p>


<p float="left">
  <img src="assets/3dpw_input1.jpg?raw=true" width="30.00%" />
  <img src="assets/3dpw_2d_input1.jpg?raw=true" width="30.00%" /> 
  <img src="assets/3dpw_mask1.jpg?raw=true" width="30.00%" />
</p>
