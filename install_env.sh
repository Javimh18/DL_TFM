conda create --name rl_env_tfm -y python=3.11.8
source activate rl_env_tfm

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gym-super-mario-bros==7.4.0
pip install tensordict==0.3.0
pip install torchrl==0.3.0

pip install matplotlib

pip install gymnasium[accept-rom-license]
pip install gymnasium[atari]
pip install timm
