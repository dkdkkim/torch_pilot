from densenet3D import densenet121
from torchsummary import summary
import torch

# USE_CUDA = torch.cuda.is_available()
# print("CUDA available:",USE_CUDA)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('num of GPU:',torch.cuda.device_count())
device = torch.device('cuda')
# Model = densenet121()
model = densenet121()
# model = UNet(dim=3)
# model.to(device)
model.cuda()
x = torch.randn(size=(1, 1, 48, 24, 48), dtype=torch.float32)
x = x.cuda()
with torch.no_grad():
    out = model(x)

print (f'Out: {out.shape}')

# summary = summary(model, (1, 48, 24, 48))