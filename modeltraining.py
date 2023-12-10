
import torch

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from data_prepocess import jai
from utilities import train





data_dir = r'D:\code\project_conqurer_hackathon\data'
model_dir = r'D:\code\project_conqurer_hackathon\data'
data_in = jai(data_dir, cache=False)
#if you wanna use gpu use true
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


#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 25, model_dir)

