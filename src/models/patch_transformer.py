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
    """
    Module that implements the patch embedding. It takes an image and divides it into patches, flattens the content and projects it into 
    another dimension using a linear transformation.
    ...
    
    Attributes
    ----------
        img_size:tuple|int
            if tuple, gives the height and the width of the input image. If int, the image shape is assumed to be equal in terms of height and width
        
    """
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
        """
        Execute a forward pass of the module.
        """
        x = self.proj(x)  # (B, E, H//P, W//P)
        # We then flatten the patches to obtain an embedding per N, where each N corresponds to the flattened (P,P)
        # It would be like erasing the spatial information of an image of PxP pixels to an a array of P^2 pixels
        x = x.flatten(2)  #(B, E, N)
        # Put the embeddings as the last dimension
        x = x.transpose(1, 2)  # (B, N, E)
        return x
        
class MultiHeadSelfAttention(nn.Module):
    """
    This class implements the Multi-head self-attention from the vision transformer. 
    ...
    
    Attributes
    ----------
    embed_dim: int
        the dimension of the embedding vector (patch channel dimension)
    n_heads: int
        the number of heads of the attention mechanism
    head_dim:int
        the dimension of the embedding when applying the attention mechanism
    Wq, Wk, Wv: torch.Tensor
        tensors (layers) that represent the weights of the query, key and value
    dropout: torch.Tensor
        layer that implements dropout with a probability p
    Wo: torch.Tensor
        tensor (layer) that projects from the attention head embedding dimension to the embedding dimension
        
    Methods
    -------
    forward(x)
        Executes the MHSA module, obtaining the attention weighted input
    """

    def __init__(self, head_dim:int, embed_dim:int, n_heads:int, dropout:float):
        """
        Initialize the attributes for the MHSA module
        
        Parameters
        ----------
            head_dim: int 
                the dimension of the projection of embedding inside the attention head
            embed_dim: int
                the dimension of the input embedding 
            n_heads: int
                number of heads attending to the input 
            dropout: float
                probability of dropout in the module
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        self.Wq = nn.Linear(self.embed_dim, self.head_dim)
        self.Wk = nn.Linear(self.embed_dim, self.head_dim)
        self.Wv = nn.Linear(self.embed_dim, self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.Wo = nn.Linear(self.head_dim, self.embed_dim)
        
    def forward(self, x):
        """
        Execute a forward pass of the module.
        """
        B, N, E = x.shape
        if self.embed_dim != E:
            print(f"MHSA: The embeding dimension does not match with the input dimension. \n Input dim: {E}, Embed Dim: {self.embed_dim}")
            exit(-1)
        
        q = self.Wq(x).reshape(B, N, self.n_heads, self.head_dim//self.n_heads)\
                      .permute(0, 2, 1, 3)# (B, N, E) * (E, H) -> (B, nh, N, H//nh)
        k = self.Wk(x).reshape(B, N, self.n_heads, self.head_dim//self.n_heads)\
                      .permute(0, 2, 1, 3)# (B, N, E) * (E, H) -> (B, nh, N, H//nh)
        v = self.Wv(x).reshape(B, N, self.n_heads, self.head_dim//self.n_heads)\
                      .permute(0, 2, 1, 3)# (B, N, E) * (E, H) -> (B, nh, N, H//nh)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(q.shape[-1]), dim=-1) # (B, nh, N, H//nh) * (B, nh, H//nh, N) -> (B, nh, N, N)
        x = (attn @ v) # (B, nh, N, N) * (B, nh, N, H//nh) -> (B, nh, N, H//nh)
        x = x.transpose(1,2) # (B, nh, N, H//nh) -> (B, N, nh, H//nh)
        x = x.reshape(B, N, self.head_dim) # (B, N, nh, H//nh) -> (B, N, H)
        x = self.dropout(x)
        x = self.Wo(x) # (B, N, H) -> (B, N, E)
        
        return x
        
        
class AttentionBlocks(nn.Module):
    """
    This class implements an Attention block, that comprises the multi-head self attention plus some
    additional layers to implement an essential block of the encoder from the transformer
    ...
    
    Attributes
    ----------
    attn_heads: int 
        number of attention heads in the attention block
    embed_dim: int
        the dimension of the input embedding 
    dropout: float
        probability of dropout in the multi-head self attention module
    head_dim: int 
        the dimension of the projection of embedding inside the attention head
    """
    def __init__(self, head_dim: int, embed_dim: int, attn_heads: int, dropout: float):
        """
        Initialize the attributes for each attention block
        
        Parameters
        ----------
            head_dim: int 
                the dimension of the projection of embedding inside the attention head
            embed_dim: int
                the dimension of the input embedding 
            n_heads: int
                number of heads attending to the input 
            dropout: float
                probability of dropout in the module
        """
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
        """
        Execute a forward pass of the module.
        """
        x = x + self.mhsa(self.ln(x)) # (B, N, E) -> (B, N, E)
        x = x + self.mlp(self.ln(x)) # (B, N, E) -> (B, N, E)
        # x = self.ln(x) # (B, N, E) -> (B, N, E)
        return x
        
        
class PatchTransformer(nn.Module):
    """
    Pytorch module that implements the patch transformer. It is an altered more flexible implementation of the Vision Transformer.
    ...
    Attributes
    ----------
        n_layers:int
            number of layers (attention blocks of the encoder) that are involved in the model.
        n_actions:int
            number of actions that the environment allows to act in the environment.
        embed_dim:int
            dimension of the patch embedding that goes into the attention module
        head_dim:int
            the dimension of the patch embedding inside the MHSA module
        attn_heads:list
            number of attention heads for each layer/attention block
        patch_size:int
            size of the projection on which the image is divided in patches (for the patch embedding)
        dropouts:list
            list with the dropouts for each attention layer block
        input_shape:tuple
            shape of the batched input tensor
        fc_dim:int
            the dimension of the fully connected linear layer that picks the attention weighted embeddings and projects 
            them onto the final action layer
        patch_embed:PatchEmbedding(nn.Module)
            module that takes the image as an input, divides it into patches, flattens its content and projects it into 
            the embed_dim dimension
        pos_embed:nn.Parameter
            a learnable parameter that serves the transformer to understand the position of the patch embeddings in the image
        model:nn.Module
            the list of the concatenated sequential attention blocks of len n_layers
        avgpool:nn.Module
            layer that takes the output attention weighted embedding and colapses the embedding dimension using an average i.e. (B, N, E) -> (B, N)
        fc:nn.Linear
            fully connected layer that takes as input the attention embeddings from the MHSA
        action_layer:nn.Linear
            layer that takes the output from the fc layer onto the action layer that gives the q_values of the possible actions
        relu: nn.ReLu
            layer that applies the rectified linear uniform activation function
            
    Methods
    -------
    forward(x)
        Executes the MHSA module, obtaining the attention weighted input
    """
    
    def __init__(self, 
                 n_layers: int, 
                 n_actions: int, 
                 patch_size: int,
                 fc_dim: int,
                 embed_dim: int, 
                 head_dim: int,
                 attn_heads: list,
                 dropouts: list,
                 input_shape: tuple):
        """
        Initialize the whole patch transformer module. IMPORTANT: It does not have a class
        token parameter, instead it averages each patch embedding and forwards it to the 
        output layer that selects which action to perform.
        
        Parameters
        ----------
            n_layers: int
                number of attention blocks that are involved into the transfomer module
            n_actions: int
                number of actions that the environment allows to perform
            embed_dim: int
                the dimension of the embedding for each layer
            head_dim: int
                the dimension for the Self Attention linear transformations
            attn_heads: list
                list that contains the attention heads for each layer (1 element -> 1 layer)
            patch_size: int
                size of the patch that is projected from the image onto the enbedding dimension
            dropouts: list
                list that contains the probabilities of dropout for each layer
            input_shape: tuple
                contains the shape of the mini-batch input that is forwarded through the net
            fc_dim: int
                the dimension of the fully connected layer that takes as input the attention embeddings from the MHSA
            
        """
        super().__init__()
        self.n_layers = n_layers 
        self.n_actions = n_actions
        self.embed_dim = embed_dim 
        self.head_dim = head_dim 
        self.attn_heads = attn_heads 
        self.patch_size = patch_size 
        self.dropouts = dropouts 
        self.input_shape = input_shape 
        self.fc_dim = fc_dim
        
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
        """
        Execute a forward pass of the module.
        """
        e = self.patch_embed(x) # (B, C, H, W) -> (B, N, E)
        e = e + self.pos_embed # adding the learned positional embedding
        
        z = self.model(e) # (B, N, E) -> (B, N, E)
        
        z = self.avgpool(z).squeeze(-1) # (B, N, E) -> (B, N)
        z = self.relu(z)
        z = self.fc(z) # (B, N) -> (B, FC)
        z = self.relu(z)
        q_values = self.action_layer(z) # (B, FC) -> (B, action_space)
        
        return q_values
        
        
         
         