# Leyenda siglas de los tensores:
# - B: Batch Size
# - E: Dimensión del embedding que contiene la información de los patches
# - P: Tamaño en alto y ancho del patch (proyección sobre la imagen original)
# - H: Altura de la imagen original
# - W: Anchura de la imagen original
# - N: La dimensión aplanada de las dimensiones del patch P
# - H: La dimensión que se tiene concatenando las cabezas de la self attention 

import torch
from torch import nn
import math

class PatchEmbedding(nn.Module):
    # Code taken from https://colab.research.google.com/github/juansensio/blog/blob/master/064_vit/vit.ipynb#scrollTo=vanilla-toolbox
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # divide the image using the integer division of the image dimensions H, W
        # and the patch dimensions, such that we get a tensor with the dimensions 
        # (Batch, Embed, H//P, W//P) 
        if isinstance(img_size, tuple) and isinstance(patch_size, tuple):
            self.n_patches = (img_size[0] // patch_size[0])*(img_size[1] // patch_size[1])
        elif not isinstance(img_size, tuple) and isinstance(patch_size, tuple) or \
            isinstance(img_size, tuple) and not isinstance(patch_size, tuple):
            print("The format of image_size and path_size must be the same. Exiting.")
            exit()
        else:  
            self.n_patches = (img_size // patch_size) ** 2
            
        # To compute the projections from the in_channels to the embedding dimension, 
        # we use the Conv2d. Since we have the patches size, we can use the Conv2d as 
        # the projection that takes as input the selected patch of the image, and produces
        # an output of the size of the embedding. Repeating this for all the patches with 
        # a conv stride of patch_size, produces the following transformation: B, C, H, W -> B, E, H//P, W//P
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H//P, W//P)
        # We then flatten the patches to obtain an embedding per N, where each N corresponds to the flattened (P,P)
        # It would be like erasing the spatial information of an image of PxP pixels to an a array of P^2 pixels
        x = x.flatten(2)  #(B, E, N)
        # Put the embeddings as the last dimension
        x = x.transpose(1, 2)  # (B, N, E)
        return x
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, head_dim, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # TODO: Change this code, since the attention is not computed propertly
        self.Wq = nn.Linear(self.embed_dim, self.head_dim)
        self.Wk = nn.Linear(self.embed_dim, self.head_dim)
        self.Wv = nn.Linear(self.embed_dim, self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.Wo = nn.Linear(self.head_dim, self.embed_dim)
        
    def forward(self, x):
        B, N, E = x.shape
        if self.embed_dim != E:
            print(f"MHSA: The embeding dimension does not match with the input dimension. \n Input dim: {E}, Embed Dim: {self.embed_dim}")
            exit(-1)
        
        q = self.Wq(x).reshape(B, self.n_heads, N, self.head_dim//self.n_heads) # (B, N, E) * (E, H) -> (B, nh, N, H//nh)
        k = self.Wk(x).reshape(B, self.n_heads, N, self.head_dim//self.n_heads) # (B, N, E) * (E, H) -> (B, nh, N, H//nh)
        v = self.Wv(x).reshape(B, self.n_heads, N, self.head_dim//self.n_heads) # (B, N, E) * (E, H) -> (B, nh, N, H//nh)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(q.shape[-1]), dim=2) # (B, nh, N, E//nh) * (B, nh, E//nh, N) -> (B, nh, N, N)
        x = (attn @ v) # (B, nh, N, N) * (B, nh, N, E//nh) -> (B, nh, N, E//nh)
        x = x.transpose(1,2) # (B, nh, N, E//nh) -> (B, N, nh, E//nh)
        x = x.reshape(B, N, self.head_dim) # (B, N, nh, E//nh) -> (B, N, E)
        x = self.dropout(x)
        x = self.Wo(x)
        
        return x
        
        
class AttentionBlocks(nn.Module):
    def __init__(self, head_dim, embed_dim, attn_heads, dropout):
        super().__init__()
        
        self.attn_heads = attn_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.head_dim = head_dim
        
        self.mhsa = MultiHeadSelfAttention(head_dim, embed_dim, attn_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        
        x = self.mhsa(x) # (B, N, E) -> (B, N, E)
        x = self.mlp(x) # (B, N, E) -> (B, N, E)
        x = self.ln(x) # (B, N, E) -> (B, N, E)
        return x
        
        
class PatchTransformer(nn.Module):
    def __init__(self, n_layers: int, 
                 n_actions: int, 
                 patch_size: int,
                 fc_dim: int,
                 embed_dim: int, 
                 head_dim: int,
                 attn_heads: list,
                 dropouts: list,
                 input_shape: tuple):
        
        super().__init__()
        self.n_layers = n_layers, # Number of layers (or blocks) that our model is going to have
        self.n_actions = n_actions # q-values of the actions our DDQN agent is going to perform
        self.embed_dim = embed_dim # The dimension of the embedding for each layer
        self.head_dim = head_dim # The dimension for the Self Attention linear transformations
        self.attn_heads = attn_heads # The attention heads for each layer
        self.patch_size = patch_size # Patch size of the image/frame
        self.dropouts = dropouts # The dropout rate to apply at the end of each layer
        self.input_shape = input_shape # the input shape of the image/frame we are going to process
        self.fc_dim = fc_dim # dimension of the FC layer
        
        if n_layers != len(attn_heads) or n_layers != len(dropouts):
                print("n_layers parameter not consistent with either embed_dims, attn_heads or dropouts ")
                exit(-1) 

        _, C, H, W = self.input_shape
        P = self.patch_size
        self.patch_embed = PatchEmbedding(img_size=(H, W),
                                          patch_size=(P, P),
                                          in_chans=C,
                                          embed_dim=self.embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, self.embed_dim))
        
        self.model = torch.nn.Sequential(*[(AttentionBlocks(
                                            self.head_dim,
                                            self.embed_dim,
                                            self.attn_heads[i],
                                            self.dropouts[i],
                                        )) for i in range(n_layers)])
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.patch_embed.n_patches, self.fc_dim)
        self.action_layer = nn.Linear(self.fc_dim, self.n_actions)
        self.relu = nn.ReLU()
    
        
    def forward(self, x):
        
        e = self.patch_embed(x) # (B, C, H, W) -> (B, N, E)
        e = e + self.pos_embed
        
        z = self.model(e) # (B, N, E) -> (B, N, E)
        
        z = self.avgpool(z).squeeze(-1) # (B, N, E) -> (B, N)
        z = self.relu(z)
        z = self.fc(z) # (B, N) -> (B, FC)
        z = self.relu(z)
        q_values = self.action_layer(z) # (B, FC) -> (B, action_space)
        
        return q_values
        
        
         
         