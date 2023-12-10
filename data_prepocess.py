import os
from glob import glob
from monai.transforms import (Compose,LoadImageD,ToTensorD,AddChanneld,SpacingD,ScaleIntensityRangeD,CropForegroundD,ResizeD,RandGaussianNoiseD,OrientationD)
from monai.data import Dataset
from monai.data import DataLoader
from monai.utils import first
import matplotlib.pyplot as plt


#for defining paths
data_dir = r'D:\code\project_conqurer_hackathon\data'

# Training data path
#sorted for loading files in sequence
train_img = sorted(glob(os.path.join(data_dir, 'TrainData', '*.nii')))
#training result path
train_labels = sorted(glob(os.path.join(data_dir, 'TrainLabels', '*.nii')))

# Testing data paths
test_img = sorted(glob(os.path.join(data_dir, 'val_data', '*.nii')))
#testing result path
test_labels = sorted(glob(os.path.join(data_dir, 'Val_labels', '*.nii')))
#for creating columns
train_files=[{"image":image_name,'label':label_name} for image_name,label_name in zip(train_img,train_labels) ]
test_files=[{"image":image_name,'label':label_name} for image_name,label_name in zip(test_img,test_labels) ]


#load the data in monai we use load image d (d for dictionary )
#transform data
#need to convert them into torch tensors
original_transform=Compose(
    [
        LoadImageD(keys=['image','label']),
        AddChanneld(keys=['image','label']),

        ToTensorD(keys=['image','label']),

    ]
)

train_transform=Compose(
    [
        LoadImageD(keys=['image','label']),
        AddChanneld(keys=['image','label']),
        SpacingD(keys=['image','label'],pixdim=(2,2,2),mode=("bilinear","nearest")),
        OrientationD(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRangeD(keys='image',a_min=-67,a_max=36,b_min=0.0,b_max=1.0,clip=True),
        CropForegroundD(keys=['image','label'],source_key='image'),
        ResizeD(keys=['image','label'],spatial_size=[128,128,64]),
        ToTensorD(keys=['image','label']),

    ]
)


test_transform = Compose(
    [
        LoadImageD(keys=['image', 'label']),
        AddChanneld(keys=['image','label']),
        SpacingD(keys=['image','label'],pixdim=(2,2,2),mode=("bilinear","nearest")),
        OrientationD(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRangeD(keys='image',a_min=-67,a_max=36,b_min=0.0,b_max=1.0,clip=True),
        CropForegroundD(keys=['image','label'],source_key='image'),
        ResizeD(keys=['image','label'],spatial_size=[128,128,64]),
        ToTensorD(keys=['image', 'label']),

    ]
)
#to load data
orig_ds=Dataset(data=train_files,transform=original_transform)
orig_loader=DataLoader(orig_ds,batch_size=1)

train_ds=Dataset(data=train_files,transform=train_transform)
train_loader=DataLoader(train_ds,batch_size=1)

test_ds=Dataset(data=test_files,transform=test_transform)
test_loader=DataLoader(test_ds,batch_size=1)

test_patient=first(train_loader)
orig_patient=first(orig_loader)





plt.subplot(1,3,1)
plt.title('original')
plt.imshow(orig_patient['image'][0,0,:,:,3],cmap="gray")

#batch size , no of channel, width,height, number of slice
plt.subplot(1,3,2)
plt.title('slice of patient')
plt.imshow(test_patient['image'][0,0,:,:,30],cmap="gray")

plt.show()
#as in 2d we had pixels in 3d we have poxels
plt.subplot(1,3,3)
plt.title('label of patient')
plt.imshow(test_patient['label'][0,0,:,:,30])
plt.show()

def jai(j,cache=False):
    return train_loader ,test_loader