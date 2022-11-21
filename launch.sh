PYPATH=/home/gbc/.conda/envs/rookie/bin
CONFIG_PATH="./config/config201_latency.yml"
DATA_PATH="./data/nasbench201/nasbench201_with_edge_flops_and_params.json"
LF=7
DEVICE=0
DEVICE2=1
NETYPE='imagenet16'
# CUDA_VISIBLE_DEVICES=0 ${PYPATH}/python train_201_latency.py \
# --config_file ${CONFIG_PATH} \
# --data_path ${DATA_PATH} \
# --save_dir "./output" \
# --seed 20211117 \
# --network_type 'cifar10' \
# --latency_factor 2

CUDA_VISIBLE_DEVICES=${DEVICE} ${PYPATH}/python train_201_latency.py --seed 1 --network_type ${NETYPE} --latency_factor ${LF} &
CUDA_VISIBLE_DEVICES=${DEVICE} ${PYPATH}/python train_201_latency.py --seed 20211117 --network_type ${NETYPE} --latency_factor ${LF} &
CUDA_VISIBLE_DEVICES=${DEVICE} ${PYPATH}/python train_201_latency.py --seed 22222222 --network_type ${NETYPE} --latency_factor ${LF} &
CUDA_VISIBLE_DEVICES=${DEVICE} ${PYPATH}/python train_201_latency.py --seed 33333333 --network_type ${NETYPE} --latency_factor ${LF} &
CUDA_VISIBLE_DEVICES=${DEVICE} ${PYPATH}/python train_201_latency.py --seed 44444444 --network_type ${NETYPE} --latency_factor ${LF} &

CUDA_VISIBLE_DEVICES=${DEVICE2} ${PYPATH}/python train_201_latency.py --seed 55555555 --network_type ${NETYPE} --latency_factor ${LF} &
CUDA_VISIBLE_DEVICES=${DEVICE2} ${PYPATH}/python train_201_latency.py --seed 66666666 --network_type ${NETYPE} --latency_factor ${LF} &
CUDA_VISIBLE_DEVICES=${DEVICE2} ${PYPATH}/python train_201_latency.py --seed 77777777 --network_type ${NETYPE} --latency_factor ${LF} &
CUDA_VISIBLE_DEVICES=${DEVICE2} ${PYPATH}/python train_201_latency.py --seed 88888888 --network_type ${NETYPE} --latency_factor ${LF} &
CUDA_VISIBLE_DEVICES=${DEVICE2} ${PYPATH}/python train_201_latency.py --seed 99999999 --network_type ${NETYPE} --latency_factor ${LF} &
