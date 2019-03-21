import os
import torch
from datasets import ALOVDataset, ILSVRC2014_DET_Dataset
from helper import (Rescale, shift_crop_training_sample,
                    crop_sample, NormalizeToTensor)
from net import GOTURN_AlexNET

import cv2
print(cv2.getBuildInformation())

args = parser.parse_args()
print(args)
batchSize = args.batch_size
kSaveModel = args.save_freq
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

bb_params={}

bb_params['lambda_shift_frac'] = 5
bb_params['lambda_scale_frac'] = 15
bb_params['min_scale'] = -0.4
bb_params['max_scale'] = 0.4
transform = NormalizeToTensor()
input_size = 224

alov = ALOVDataset(os.path.join("../pygoturn/data",
                                'imagedata++/'),
                   os.path.join("../pygoturn/data",
                                'alov300++_rectangleAnnotation_full/'),
                   NormalizeToTensor(), input_size)

# imagenet = ILSVRC2014_DET_Dataset(os.path.join("../pygoturn/data",
#                                                'ILSVRC2014_DET_train/'),
#                                   os.path.join("../pygoturn/data",
#                                                'ILSVRC2014_DET_bbox_train/'),
#                                   bb_params,
#                                   transform,
#                                   input_size)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

net = GOTURN_AlexNET()

iterations = 10000
for i in range(iterations):
    alov.show_sample_no_wait(i)
    sample, _ = alov.get_sample(i)
    sample = transform(sample)
    x_curr = sample['currimg'].unsqueeze(0)
    x_prev = sample['previmg'].unsqueeze(0)
    y = sample['currbb']
    output = net((x_prev, x_curr))
    print(output, y)
