import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import argparse
from pycocotools.coco import COCO
from pycocotools import mask
import json
from tqdm import tqdm
import os.path as osp

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, dest="source", help='image file path')
    parser.add_argument('--annot_path', type=str, dest="target", help='annot file path')
    parser.add_argument('--dataset_name', type=str, dest="dataset", help='dataset',choices=["MPII","HR-LSP","CrowdPose","MSCOCO","OCHuman","MuCo","Human36M","3DPW"])
    parser.add_argument('--save_name', type=str, dest="save",default="LSP", help='target (GT SMPL) json file path')
    parser.add_argument('--model_path', type=str, dest="m_path", help='model_path')

    args = parser.parse_args()

    return args

def choose_dataset(args):
    if args.dataset=="HR-LSP":
        import glob
        import scipy.io as scio
        import os
        joints = scio.loadmat(os.path.join(args.target,'joints.mat'))['joints'].transpose(2,0,1).astype(np.float32)
        img_paths = sorted(glob.glob(os.path.join(args.source, '*.png')))
        return img_paths,joints,[]
    elif args.dataset=="OCHuman":
        """
        OCHuman 3-channel Means
        'vis': 1, 'others_occluded': 3, 'self_occluded': 2, 'missing': 0
        """
        img_paths=[]
        joints=[]
        list_id=[]
        with open(osp.join(args.target,"ochuman.json"),"r") as f:
            json_data=json.load(f)
        for annot in json_data['images']:
            img_name=annot['file_name']
            image_path=osp.join(args.source,img_name)
            img_joints=[]
            img_id=[]
            for id,person_k in enumerate(annot['annotations']):
                #try:
                if person_k['keypoints'] is None:
                    continue
                joint=np.array(person_k['keypoints']).reshape(-1,3)
                img_joints.append(joint)
                img_id.append(id)
                #except:
                #    print(person_k)
            img_paths.append(image_path)
            joints.append(img_joints)
            list_id.append(img_id)
        return img_paths,joints,list_id


    elif args.dataset=="MSCOCO" or args.dataset=="MuCo" or args.dataset=="3DPW":
        """
        MSCOCO 3-channel Means
        0 : invisible, 1 : occlusion, 2 : visible
        """
        from pycocotools.coco import COCO
        img_paths=[]
        joints=[]
        list_id=[]
        if args.dataset=="MSCOCO":
            db=COCO(osp.join(args.target,"person_keypoints_train2017.json"))
        elif args.dataset=="MuCo":
            db = COCO(osp.join(args.target, 'MuCo-3DHP.json'))
        elif args.dataset=="3DPW":
            db = COCO(osp.join(args.target, '3DPW_latest_train.json'))
        for iid in db.imgs.keys():
            aids = db.getAnnIds([iid])
            img_joints=[]
            list_id.append(aids)
            for aid in aids:
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                if args.dataset=="MSCOCO":
                    img_path = osp.join(args.source, img['file_name'])
                    joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                elif args.dataset=="MuCo":
                    img_path = osp.join(args.source, img['file_name'])
                    joint_img = np.array(ann['keypoints_img'], dtype=np.float32).reshape(-1, 2)
                    j_valid=np.ones((21,1))
                    joint_img=np.concatenate((joint_img,j_valid),axis=1)
                elif args.dataset=="3DPW":
                    sequence_name = img['sequence']
                    img_path = osp.join(args.source,sequence_name, img['file_name'])
                    joint_img = np.array(ann['joint_img'], dtype=np.float32).reshape(-1, 2)
                    j_valid=np.ones((24,1))
                    joint_img=np.concatenate((joint_img,j_valid),axis=1)
                img_joints.append(joint_img)
            if not aids:
                continue
            img_paths.append(img_path)
            joints.append(img_joints)
        return img_paths,joints,list_id

        
    elif args.dataset=="MPII" or args.dataset=="CrowdPose":
        from pycocotools.coco import COCO
        img_paths=[]
        joints=[]
        list_id=[]
        if args.dataset=="MPII":
            db=COCO(osp.join(args.target,"train.json"))
        if args.dataset=="CrowdPose":
            db=COCO(osp.join(args.target,'crowdpose_train.json'))
            
        for iid in db.imgs.keys():
            aids = db.getAnnIds([iid])
            img_joints=[]
            list_id.append(aids)
            for aid in aids:
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                img_path = osp.join(args.source, img['file_name'])
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                img_joints.append(joint_img)
            img_paths.append(img_path)
            joints.append(img_joints)
        return img_paths,joints,list_id
    #elif args.dataset=="OCHuman":

    
def main(args):

    sam_checkpoint = args.m_path
    model_type = sam_checkpoint[4:9]
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    img_paths,annots,list_id=choose_dataset(args)

    result_data={}
    if not list_id:
        for img_path,annot in tqdm(zip(img_paths,annots), total=len(img_paths)):
            # load image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            img_name=img_path.split("/")[-1]

            # load pose
            origin_pose=annot.copy()
            annot[:,2]=1
            input_point = annot[:,:2]
            input_label = annot[:,2]
            masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
            )

            # Convert
            mscoco_RLE=mask.encode(np.asfortranarray(masks[2]))
            mscoco_RLE['counts']=mscoco_RLE['counts'].decode('utf-8')
            result_data[img_name]={
                "2d_pose":origin_pose.tolist(),
                "Seg":mscoco_RLE}
            
        with open("{}_seg.json".format(args.save),"w") as f:
            json.dump(result_data,f)
    else:
        for img_path,annot, ids in tqdm(zip(img_paths,annots,list_id),total=len(img_paths)):
            # load image
            # print(img_path)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            img_name=osp.join(img_path.split("/")[-2],img_path.split("/")[-1])
            for joint, id in zip(annot,ids):
                #if not joint:
                #    continue
                if np.sum(joint[:,2])==0:
                    continue
                # SAM prompt format
                new_joint=[]
                if args.dataset!="MSCOCO":
                    for i in joint:
                        if i[2]!=0:
                            new_joint.append(i)
                elif args.dataset=="MSCOCO":
                    for i in joint:
                        if i[2]!=0 and i[2]==2:
                            new_joint.append(i)
                elif args.dataset=="OCHuman":
                    for i in joint:
                        if i[2]!=0 and i[2]!=3:
                            new_joint.append(i)
                if not new_joint:
                    continue
                new_joint=np.array(new_joint)
                new_joint[:,2]=1
                input_point = new_joint[:,:2]
                input_label = new_joint[:,2]
                masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,)
                # Convert
                mscoco_RLE=mask.encode(np.asfortranarray(masks[2]))
                mscoco_RLE['counts']=mscoco_RLE['counts'].decode('utf-8')
                result_data[img_name]={id:{
                    "2d_pose":joint.tolist(),
                    "Seg":mscoco_RLE}}
            
            with open("{}_seg.json".format(args.save),"w") as f:
                json.dump(result_data,f)
            

if __name__=="__main__":
    args = parse_args()
    main(args)