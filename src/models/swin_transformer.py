import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class WindowAttention(nn.Module):
    
    def __init__(self, embed_in_dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.embed_in_dim = embed_in_dim
        self.window_size = window_size # Wh, Ww
        self.num_heads = num_heads
        head_dim = embed_in_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        # We are going to define a relative position index, instead of an absolute position
        # index, that was defined in the ViT. We are going to define a parameter that is 
        # going to be the relative positional embedding. This parameter will give relative
        # positional information to each of the patches inside the window. To index this, 
        # we are going to use an array of (2*M-1)x(2*M-1)xnh, where nh is the number of heads.
        # This kind of indexing is better than using (M^2)x(M^2), since the relative indexing 
        # between the positions can be parametrized as 2*M-1 values for M^2 possibilities.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1) * (2*window_size[1]-1), num_heads)
        )
        
        # TODO: Entender el relative positional embedding antes de seguir escribiendo el 
        # modelo.

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)
    
    def forward(self, x):
        P_H, P_W = self.input_resolution
        B, N, E = x.shape
        
        x = x.view(B, P_H, P_W, E)
        
        # Reducing the height and width dimensions of the vectors by a factor of 2.
        # In x0 we will store the patches that are even (0,2,4...) for both H & W
        # In x1 we will store the pathes that are even in the width dimension but odd
        # in the height dimension. In x2 we will do the viceversa process, and finally,
        # in x3 we will only get the patches that are un odd positions both for the height
        # and width coordinates. The resulting dimensions for each patch will be (B P_H/2 P_W/2 C)
        x0 = x[:, 0::2, 0::2, :] 
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        # Concatenating them in the -1 dimension (the last embedding) we obtain
        # the following tensor dimensions (B P_H/2 P_W/2 4*C)
        x = torch.cat([x0, x1, x2, x3], -1)
        # now, since we dont need the spatial dimensions P_H and P_W, we can merge them
        # into an tensor of the dimensions (B, P_H/2*P_W/2, 4*C)
        x = x.view(B, -1, 4 * E)
        # Now we apply LayerNorm to the tensor, so the embeddings are normalized.
        # Since now the embeddings are four times larger (but reduced in the spatial 
        # dimension), the norm_layer is declared in the constructor of the class as
        # norm_layer(4*dim)
        x = self.norm(x)
        # Now we reduce the embedding dimension from 4*embed_dim to 2*embed_dim
        # making a linear projection as a dense layer
        x = self.reduction(x)
        
        return x
        
    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

class SwinTransformerBlock(nn.Module):
    
    def __init__(self, embed_in_dim, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.embed_in_dim = embed_in_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # quick check. If window size is larger than input resolution,
        # we don't partition the patched image using windows
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(embed_in_dim)
        
        # TODO: Entender el modulo de WindowAttention antes de seguir con el transformer block
        self.attn = WindowAttention(
            
        )
        

class BasicLayer(nn.Module):
    def __init__(self, embed_in_dim, input_resolution, depth,
                 num_heads, window_size, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.embed_in_dim = embed_in_dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # we are going to build the blocks that participate in a layer of 
        # a Swin Transformer. Each SwinTransformer block will have the essential
        # computation block that produces the embeddings of the input. Those embeddings
        # may either be passed to another block or used as the feature embedding for
        # the next block to use as an input
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(embed_in_dim=self.embed_in_dim,
                                 input_resolution = self.input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, 
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        
        # Apply (or not) patch merging depending on the depth of the layer
        # inside the stage. In the odd stages, we do not apply patch merging,
        # in the even, we do
        if downsample is not None:
            self.downsample = downsample(input_resolution, embed_in_dim, norm_layer)
        else:
            self.downsample = None
            
    def extra_repr(self) -> str:
        return f"dim={self.embed_in_dim}, input_resolution={self.input_resolution}, depth={self.depth}"
            
    def forward(self, x):
        
        # iterate over the blocks, SwinTransformerBlock, and forward the output
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
            
            
class PatchEmbed(nn.Module):
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        # this gets the number of patches that fit into the image,
        # given the patch size that we've set
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0]*patches_resolution[0]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # we use the conv2d as a projection function that maps from
        # the original B, C, H, W shape of the input tensor to a 
        # B, E, H//P, W//P tensor, where "E" is the embedding dimension
        self.proj = nn.Conv2d(in_channels=self.in_chans, 
                              out_channels=self.embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size)
        # if normalization layer set, then apply it
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
            
    def forward(self, x):
        # Apply the following transformations:
        # 1. proj (B,C,H,W) -> (B, E, H//P, W//P)
        # 2. flatten (B, E, H//P, W//P) -> (B,E,H//P*W//P)
        # 3. transpose (B,E,H//P*W//P) -> (B,H//P*W//P,E)
        x = self.proj(x) \
                .flatten(2) \
                .transpose(1,2)
                
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class SwinTransformer(nn.Module):
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 num_classes=1000, embed_dim=96, depths=[2,2,6,2],
                 num_heads=[3,6,12,24], window_size=7, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, 
                 path_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = path_norm
        # TODO: Why use this parameter to compute the number of features for the 
        # final FC layer?
        self.num_features = int(embed_dim*2 ** (self.num_layers-1))
        
        # The ratio of expansion of the hidden dimension inside the MLP that 
        # uses as input the output of the S-MSA or SW-MSA.
        self.mlp_ratio = mlp_ratio
        
        # split image into patches that do not overlap. This is to use 
        # patches as the atomic unit that embeds the informatio of the image
        # the windowing will be done by projecting the windows over the patches.
        # i.e. a window of 7x7 will contain 49 patches in it. A patch will contain the 
        # the embedding, that is a linear projection from the original dimension of the
        # pixels contained in the patch (i.e 4x4) to an embedding dimension "E"
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # absolute position embedding. The position embedding is at 
        # patch level. Each patch has its own positional embedding
        # and when the windowing its applied, this positional embedding
        # will give information not only of the relative position within the
        # window, but also the global position of where that patch is. The embedding 
        # is learnt in the training process, and the main idea is that the model itself
        # will learn which patch goes where for a better scene understanding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1,
                                                               num_patches,
                                                               embed_dim))
            # TODO: It applies the normal distribution to the positional
            # encoding tensor, but, why do that? It to ponder the information 
            # of the dimensions of the embeddings?
            trunc_normal_(self.absolute_pos_embed, std=.02)
            
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # stochastic depth for training really deep neural nets. When the depth
        # of the network starts to increase, problems like vanishing gradients
        # arise. One of the approaches that may result in better results is to
        # associate, for each layer, a probability that the layer is not involved 
        # in the training process. As layers go deeper, more probability should be
        # assigned.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # building the layers of the SWIN Transformer. We assume that each
        # basic layer is an stage of the architecture shown in the original paper.
        # The embedding of the patches will be deeper each stage (meaning that the
        # dimension will be bigger), and can be modeled using the following formula
        # embed_dim*(2**i_layer)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                # for each layer, as the spatial information gets lower, 
                # the embed_dim is inflated by two each layer
                embed_in_dim=int(embed_dim*(2**i_layer)), 
                input_resolution=(patches_resolution[0]//(2**i_layer),
                                  patches_resolution[1]//(2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
            
        self.norm = norm_layer(self.num_features)
        # this layer averages the N dimension from a (B, E, N) type of 
        # tensor, resulting in a (B,E,1) tensor that averages all the values 
        # from the N dimension into 1 number.
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # project from the embed dimension to the output dimension
        self.head = nn.Linear(self.num_features, num_classes) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        # get the patches from the images and extract the embeddings for each 
        # patch
        x = self.patch_embed(x)
        # if absolute positional encoding is specified, apply it to the
        # input of the layer
        if self.ape:
            x = x + self.absolute_pos_embed
        # apply dropout to the connections that go from the patch_embed encoded input
        # to the layers from the transformer
        x = self.pos_drop(x)
        
        # apply the basic_layer blocks (stages)
        for layer in self.layers:
            x = layer(x)
            
        # apply layer norm to the embeddings that produce the stages 
        # (stacked BasicLayers)
        x = self.norm(x) # B, N, E
        
        # Apply the avg of the values in the N dimension to obtain
        # and embedding vector that agreggates all the spatial informatio
        x = self.avgpool(x.transpose(1,2)) # (B, E, N) -> (B, E, 1)
        # Squeeze (or flatten) the last dimension to obtain a (B,E) tensor
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
            
        