import params
class Cluster(object):
    def __init__(self):
   
        
        self.cluster_num_cpu = None
        self.cluster_num_mem = None
        self.cluster_num_gpu = None
        
        self._set_cluster_config()

        

        #节点资源使用状况，已用数量
        self.node_used_cpu_list = {}   #[time][r]   time时刻r节点的已用资源数量
        self.node_used_mem_list = {}
        self.node_used_gpu_list = {}
        self.node_used_bw_list ={}     #[time][r][r']   time时刻 r和r'之间的已用带宽，每个key对应二维数组（对称矩阵）
        
        for x in range(1,params.T+1):
            self.node_used_cpu_list[x]=[0 for i in range(len(params.NODE_LIST))]
            self.node_used_mem_list[x] = [0 for i in range(len(params.NODE_LIST))]
            self.node_used_gpu_list[x] = [0 for i in range(len(params.NODE_LIST))]
            self.node_used_bw_list[x] = [[0 for i in range(len(params.NODE_LIST))] for i in range(len(params.NODE_LIST))]


    def _set_cluster_config(self):
        cluster_num_nodes = len(params.NODE_LIST)
        cpu_per_node = params.CPU_PER_NODE
        mem_per_node = params.MEM_PER_NODE
        #bw_per_node = params.BW_PER_NODE
        gpu_per_node = params.GPU_PER_NODE
        self.cluster_num_cpu = cluster_num_nodes * cpu_per_node
        self.cluster_num_mem = cluster_num_nodes * mem_per_node
        self.cluster_num_gpu = cluster_num_nodes * gpu_per_node
        
        #self.cluster_num_bw =  bw_per_node
        #self.cluster_cost_bw = bw_per_node_cost