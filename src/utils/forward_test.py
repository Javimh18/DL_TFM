import torch
from torch import nn
from src.models.patch_transformer import PatchTransformer
from torchsummary import summary    



if __name__ == '__main__':
    B = 8
    H = 64
    W = 64
    C = 4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randint(low=0, high=10, size=(B, C, H, W)).float()
    model = PatchTransformer(n_layers=2,
                           n_actions=4,
                           patch_size=8,
                           fc_dim=16,
                           embed_dim=128,
                           attn_heads=[4,8],
                           dropouts=[0.3, 0.3],
                           input_shape=x.shape)
    model = model.to(device)
    
    summary(model, (C, H, W))