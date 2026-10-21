import torch
from src.models import OfficialCNN,MSCoordNet

def main():
    a=OfficialCNN(); y,_=a(torch.randn(8,120,3)); assert y.shape==(8,)
    b=MSCoordNet(); y,z=b(torch.randn(8,120,24),torch.ones(8,120,24)); assert y.shape==(8,); assert z['activity_logits'].shape==(8,8)
    print('smoke test passed')
if __name__=='__main__': main()
