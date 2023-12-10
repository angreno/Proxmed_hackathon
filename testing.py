from monai.utils import first
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import  DataLoader, Dataset

import torch
import matplotlib.pyplot as plt

import os
from glob import glob
import numpy as np

from monai.inferers import sliding_window_inference
in_dir = 'D:\code\project_conqurer_hackathon\data'
model_dir = 'D:\code\project_conqurer_hackathon\data'

train_loss = np.load(os.path.join(model_dir, 'loss_train2.npy'))
train_metric = np.load(os.path.join(model_dir, 'metric_train2.npy'))
test_loss = np.load(os.path.join(model_dir, 'loss_test2.npy'))
test_metric = np.load(os.path.join(model_dir, 'metric_test2.npy'))

plt.figure("Results 25 june", (12, 6))
plt.subplot(2, 2, 1)
plt.title("Train dice loss")
x = [i + 1 for i in range(len(train_loss))]
y = train_loss
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.title("Train metric DICE")
x = [i + 1 for i in range(len(train_metric))]
y = train_metric
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 3)
plt.title("Test dice loss")
x = [i + 1 for i in range(len(test_loss))]
y = test_loss
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 4)
plt.title("Test metric DICE")
x = [i + 1 for i in range(len(test_metric))]
y = test_metric
plt.xlabel("epoch")
plt.plot(x, y)

plt.show()


path_train_data = sorted(glob(os.path.join(in_dir, "TrainData", "*.nii")))
path_train_label = sorted(glob(os.path.join(in_dir, "TrainLabels", "*.nii")))

path_test_data= sorted(glob(os.path.join(in_dir, "val_data", "*.nii")))
path_test_label = sorted(glob(os.path.join(in_dir, "Val_labels", "*.nii")))

train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(path_train_data, path_train_label)]
test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(path_test_data, path_test_label)]
test_files = test_files[4:9]

test_transform = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image','label']),
        Spacingd(keys=['image','label'],pixdim=(2,2,2),mode=("bilinear","nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys='image',a_min=-67,a_max=36,b_min=0.0,b_max=1.0,clip=True),
        CropForegroundd(keys=['image','label'],source_key='image'),
        Resized(keys=['image','label'],spatial_size=[64,64,64]),
        ToTensord(keys=['image', 'label']),

    ]
)
test_ds = Dataset(data=test_files, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=1)

#mode
device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model.pth2")))
model.eval()

sw_batch_size = 4
roi_size = (128, 128, 64)
with torch.no_grad():
    test_patient = first(test_loader)
    t_volume = test_patient['image']
    t_segmentation = test_patient['label']

    test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
    sigmoid_activation = Activations(sigmoid=True)
    test_outputs = sigmoid_activation(test_outputs)
    test_outputs = test_outputs > 0.53

    for i in range(30):
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {80}")
        plt.imshow(test_patient["image"][0, 0, :, :, 30], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {80}")
        plt.imshow(test_patient["label"][0, 0, :, :, 30] != 0)
        plt.subplot(1, 3, 3)
        plt.title(f"output {80}")
        plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, 30] !=0)
        plt.show()