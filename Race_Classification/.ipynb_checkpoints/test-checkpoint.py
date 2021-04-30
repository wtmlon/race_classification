from network.res_se_net import se_resnet_18
from network.resnet import resnet18
import torch
model = se_resnet_18()
model1 = resnet18()
a = torch.FloatTensor(torch.randn([1, 3, 178, 218]))
out = model(a)
print(out)