import numpy as np

JOB_ARRIVAL = "uniform"
JOB_DISTRIBUTION = "uniform"
RANDOM_SEED = 9973
POISSON_LAMBDA = 3


BW_PER_NODE = 10000
CPU_PER_NODE = 64
GPU_PER_NODE = 8
MEM_PER_NODE = 64

TOT_NUM_JOBS = 2
T = 50
B = 10000
base_B = 2
base_res = [2,2,2]

#JOB_SCHEDULER = "FIFO"
#JOB_SCHEDULER = "DRF"
#JOB_SCHEDULER = "Antman"
#JOB_SCHEDULER = "Liquid"
#JOB_SCHEDULER = "Tiresias"
JOB_SCHEDULER = "Eris"


LOG_LEVEL = "DEBUG"
MIN_SLEEP_UNIT = 10**-3

TS_INTERVAL = 15 #seconds
DEFAULT_NUM_PS = 1
DEFAULT_NUM_WORKER = 1

# cluster node ips+
NODE_LIST = ['pjl-master','wxy-node2','zrl-node3','zy-node1','wrb-node4','dell-node5','dell-node5','node6','node7'] #
LOSS_LITTLE_CHANGE_EPOCH_NUM = 10
LOSS_CONVERGENCE = 0.05

