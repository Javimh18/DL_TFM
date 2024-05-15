conda create --name rl_env_tfm -y python=3.11.8
source activate rl_env_tfm

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install gym-super-mario-bros==7.4.0
pip3 install tensordict==0.3.0
pip3 install torchrl==0.3.0

pip3 install matplotlib

pip3 install gymnasium[accept-rom-license]
pip3 install gymnasium[atari]
pip3 install timm
pip3 install moviepy
pip3 install torchsummary
pip3 install pyyaml