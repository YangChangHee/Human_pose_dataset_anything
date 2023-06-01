# Human pose dataset anything (segmentation pseudo GT)


We have created this repository to share code using SAM (Segmentation Anything) for generating pseudo ground truth (GT) for individuals using human pose datasets.

## Tasks to be done
* Make CrowdPose & Human36M & 3DOH50K pseudo GT 
* Make Visualization Code
* Upload Pseudo GT annotation json file

## Other dataset
If you need pseudo GT annotations for other datasets, please mention it in the issue.

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

## Download pseudo GT
LSP, MPII, HR-LSPET, MSCOCO, OCHuman

[google-drive](https://drive.google.com/drive/folders/1z-rHkgSEGFVD3QysYzX2ytrCH5E2gnku?usp=sharing)

## Getting started
The directory structure of our data is as follows:
```
| data
| - Human36M
| - - images
| - - annotations
| - MuCo
| - - data
| - - - augmented_set
| - - - unaugmented_set
| - - MuCo-3DHP.json
| - OCHuman
| - - images
| - - ochuman.json
| - 3dpw
| - - data
| - - - 3DPW_latest_train.josn
| - - imageFiles
| - HR-LSPET
| - - hr-lsped
| - - - joints.mat
| - - - image.jpg...
| - MPII
| - - images
| - - annotations
| - MSCOCO
| - - images
| - - - train2017
| - - annotations
| - - - person_keypoints_train2017.json
| - CrowdPose
| - - images
| - - annotations
```

### Write the following code in the terminal to generate pseudo GT.
### LSP
```
python segment_anything_pseudo.py --image_path {data_path}/HR-LSPET/hr-lspet --annot_path {data_path}/HR-LSPET/hr-lspet --dataset_name HR-LSP --save_name {save_name} --model_path sam_vit_h_6b8939.pth
```

### MPII
```
python segment_anything_pseudo.py --image_path {data_path}/MPII/images --annot_path {data_path}/MPII/annotations --dataset_name MPII --save_name {save_name} --model_path sam_vit_h_6b8939.pth
```

### MSCOCO
```
python segment_anything_pseudo.py --image_path {data_path}/MSCOCO/images/train2017 --annot_path {data_path}/MSCOCO/annotations --dataset_name MSCOCO --save_name {save_name} --model_path sam_vit_h_6b8939.pth
```

### OCHuman
```
python segment_anything_pseudo.py --image_path {data_path}/OCHuman/images/ --annot_path {data_path}/OCHuman/ --dataset_name OCHuman --save_name {save_name} --model_path sam_vit_h_6b8939.pth
```


### CrowdPose
We plan to make modifications to the code in the future as it is not currently demonstrating satisfactory performance
```
python segment_anything_pseudo.py --image_path {data_path}/CrowdPose/images/ --annot_path {data_path}/CrowdPose/annotations --dataset_name OCHuman --save_name {save_name} --model_path sam_vit_h_6b8939.pth
```


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
