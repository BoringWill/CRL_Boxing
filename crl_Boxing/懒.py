su - chenghan233
cd /mnt/c/Users/asus/Desktop/crl_Boxing
conda activate rl_env
tensorboard --logdir=runs
killall - 9 python
export DISPLAY =$(ip route show | grep -i default | awk '{print $3}'): 0

sudo apt-get update
sudo apt-get install -y zlib1g-dev libsdl2-dev cmake
AutoROM --accept-license
torchrun --standalone --nnodes=1 --nproc_per_node=4 main_ddp.py #ddp
torchrun --nproc_per_node=4 trueskill_rank_model.py --opponent-pool-path ./历史模型最新  #rank