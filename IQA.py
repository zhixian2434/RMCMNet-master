import pyiqa
import torch
from glob import glob

# list all available metrics
print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create metric with default setting
iqa_metric = pyiqa.create_metric('niqe', device=device)

# check if lower better or higher better
#print(iqa_metric.lower_better)


# img path as inputs.
test_list = glob("/home/liu/wzl/RetinexMac/result/VV/*")
niqes = 0
for image in test_list:
    score_fr = iqa_metric(image)
    niqes+=score_fr
print("NIQE: %.3f " % (niqes / len(test_list)))

