import torch
from torch import nn

from models.patch_transformer import PatchTransformer
from models.vit import ViT
from models.cnn import CNN

class DQN(nn.Module):
    def __init__(self, type:str, n_actions:int, obs_shape:tuple, config:dict) -> None:
        super().__init__()
        nn_config = config[type]
        if type == 'vit':
            P = int(nn_config['patch_size'])
            self.online = ViT(img_size=obs_shape,
                              patch_size=(P, P),
                              embed_dim=int(nn_config['embed_dim']),
                              n_heads=int(nn_config['n_heads']),
                              n_layers=int(nn_config['n_layers']),
                              n_actions=n_actions)
            
            self.target = ViT(img_size=obs_shape,
                              patch_size=(P, P),
                              embed_dim=int(nn_config['embed_dim']),
                              n_heads=int(nn_config['n_heads']),
                              n_layers=int(nn_config['n_layers']),
                              n_actions=n_actions)
        elif type == 'swin':
            # TODO: Inicializar SWIN Transformer...
            pass
        elif type == 'patch_transformer':
            self.online = PatchTransformer(n_actions=n_actions,
                                n_layers=int(nn_config['n_layers']),
                                patch_size=int(nn_config['patch_size']),
                                fc_dim=int(nn_config['fc_dim']),
                                head_dim=int(nn_config['head_dim']),
                                embed_dim=int(nn_config['embed_dim']),
                                attn_heads=nn_config['attn_heads'],
                                dropouts=nn_config['dropouts'],
                                input_shape=(1,)+obs_shape)
            
            self.target = PatchTransformer(n_actions=n_actions,
                               n_layers=int(nn_config['n_layers']),
                                patch_size=int(nn_config['patch_size']),
                                fc_dim=int(nn_config['fc_dim']),
                                head_dim=int(nn_config['head_dim']),
                                embed_dim=int(nn_config['embed_dim']),
                                attn_heads=nn_config['attn_heads'],
                                dropouts=nn_config['dropouts'],
                                input_shape=(1,)+obs_shape)
        elif type == 'cnn':
            self.online = CNN(n_actions=n_actions)
            self.target = CNN(n_actions=n_actions)
        else:
            print(f"Type of agent for the model is not correctly specified.\n"
                  f"Select between:\n"
                  f"\t - Visual Transformer (vit)\n"
                  f"\t - Patch Transformer (patch_transformer)\n"
                  f"\t - SWIN Transformer (swin)\n"
                  f"Exiting..."
                  )
            exit()
            
        self.target.load_state_dict(self.online.state_dict())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Q_online network parameters must be trained
        for p in self.online.parameters():
            p.requires_grad = True
            
        # Q_target network parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        