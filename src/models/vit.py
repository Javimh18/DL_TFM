# Code taken from https://colab.research.google.com/github/juansensio/blog/blob/master/064_vit/vit.ipynb#scrollTo=vanilla-toolbox

# Leyenda siglas de los tensores:
# - B: Batch Size
# - E: Dimensión del embedding que contiene la información de los patches
# - P: Tamaño en alto y ancho del patch (proyección sobre la imagen original)
# - H: Altura de la imagen original
# - W: Anchura de la imagen original
# - N: La dimensión aplanada de las dimensiones del patch P

import torch
from torch import nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        C, H, W = img_size
        self.img_size = (H, W)
        self.patch_size = patch_size
        
        # divide the image using the integer division of the image dimensions H, W
        # and the patch dimensions, such that we get a tensor with the dimensions 
        # (Batch, Embed, H//P, W//P)
        if isinstance(img_size, tuple) and isinstance(patch_size, tuple):
            self.n_patches = (self.img_size[0] // self.patch_size[0])*(self.img_size[1] // self.patch_size[1])
        elif not isinstance(self.img_size, tuple) and isinstance(self.patch_size, tuple) or \
            isinstance(self.img_size, tuple) and not isinstance(self.patch_size, tuple):
            print("The format of image_size and path_size must be the same. Exiting.")
            exit()
        else:   
            self.n_patches = (img_size // patch_size) ** 2
            
        # To compute the projections from the in_channels to the embedding dimension, 
        # we use the Conv2d. Since we have the patches size, we can use the Conv2d as 
        # the projection that takes as input the selected patch of the image, and produces
        # an output of the size of the embedding. Repeating this for all the patches with 
        # a conv stride of patch_size, produces the following transformation: B, C, H, W -> B, E, H//P, W//P
        self.proj = nn.Conv2d(C, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H//P, W//P)
        # We then flatten the patches to obtain an embedding per N, where each N corresponds to the flattened (P,P)
        # It would be like erasing the spatial information of an image of PxP pixels to an a array of P^2 pixels
        x = x.flatten(2)  #(B, E, N)
        # Put the embeddings as the last dimension
        x = x.transpose(1, 2)  # (B, N, E)
        return x
    
class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads 
        self.embed_dim = n_embd
        
        # key, query, value projections
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        B, N, E = x.size()
        
        if E != self.n_heads: # The dimension of the embedding should not change
            print(f"MHSA: The embeding dimension does not match with the input dimension. \n Input dim: {E}, Embed Dim: {self.embed_dim}")
            exit(-1)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, E//self.n_heads).permute(2, 0, 3, 1, 4) # (B, N, 3*E) -> (B, N, 3, nh, E//nh) -> (3, B, nh, N, E//nh)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        # attention (B, nh, N, E//nh) x (B, nh, E//nh, N) -> (B, nh, N, N) 
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v # (B, nh, N, N) x (B, nh, N, E) -> (B, nh, N, E)
        y = y.transpose(1, 2).contiguous().view(B, N, E) # re-assemble all head outputs side by side
        
        return self.drop(y)
    
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class ViT(nn.Module):

    def __init__(self, img_size=(28,28), patch_size=7, embed_dim=100, n_heads=3, n_layers=3, n_actions=10):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        self.action_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        
        self.tranformer = torch.nn.Sequential(*[TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)])
        
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = torch.nn.Linear(embed_dim, n_actions)

    def forward(self, x):
        e = self.patch_embed(x)
        B, N, E = e.size()
        
        action_token = self.action_token.expand(B, -1, -1)  # Expand it to the batch dimension for batch operationsn (B, 1, E)
        e = torch.cat((action_token, e), dim=1)  # Add the the class token to the patch embedding (B, 1 + N, E)
        e = e + self.pos_embed  # To each token, add the positional encoding for semantic understanding of the scene (B, 1 + N, E)
        
        z = self.tranformer(e) # Get the output embedding with the aggregated feature information from the image
        
        action_token_final = z[:, 0] # Not sure about this... maybe use the whole attention map and not use the action_token?
        y = self.fc(action_token_final)

        return y
