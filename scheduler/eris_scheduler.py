from __future__ import division
import copy
import Queue
import os
import time
import sys
import threading
import math
import params
import random
from estimator import Estimator
#from cluster import Cluster

class Eris_Scheduler(object):
    def __init__(self, name, logger, scheduler_queue, hub_queue, timer):
        self.name = name  # e.g., 'UTIL'
        self.logger = logger
        self.scheduler_queue = scheduler_queue
        self.hub_queue = hub_queue
        self.timer = timer

        self.cluster_num_cpu = None
        self.cluster_num_mem = None
        self.cluster_num_gpu = None
        
        self.cluster_used_cpu = 0
        self.cluster_used_mem = 0
        self.cluster_used_gpu = 0
        self.cluster_used_bw = 0
        self._set_cluster_config()

        cluster_num_nodes = len(params.NODE_LIST)
        cpu_per_node = params.CPU_PER_NODE
        mem_per_node = params.MEM_PER_NODE
        #bw_per_node = params.BW_PER_NODE
        gpu_per_node = params.GPU_PER_NODE
        self.cluster_num_cpu = cluster_num_nodes * cpu_per_node
        self.cluster_num_mem = cluster_num_nodes * mem_per_node
        self.cluster_num_gpu = cluster_num_nodes * gpu_per_node
        self.cluster_num_bw = params.B
        self.cluster_B = params.B


        # resource usage
        self.node_used_cpu_list = {}   #[time][r]
        self.node_used_mem_list = {}
        self.node_used_gpu_list = {}
        self.used_b = {}     #[time]

        # resource price
        self.price_B = {}  # [time]
        self.price_cpu = {}  # [time][r]
        self.price_mem = {}
        self.price_gpu = {}

        for x in range(1,params.T+1):#10 extra time slot for overflow
            self.node_used_cpu_list[x] = [0 for i in range(len(params.NODE_LIST))]
            self.node_used_mem_list[x] = [0 for i in range(len(params.NODE_LIST))]
            self.node_used_gpu_list[x] = [0 for i in range(len(params.NODE_LIST))]
            self.used_b[x] = 0

            self.price_B[x] = 0
            self.price_cpu[x] = {}
            self.price_mem[x] = {}
            self.price_gpu[x] = {}

        self._price_update()

        self.social_welfare = 0
        
        # the payoff of the service provider
        self.payoff = 0

        self.estimator = Estimator("estimator", self.logger)

        self.queueing_jobs = Queue.PriorityQueue()
        self.uncompleted_jobs = []
        self.completed_jobs = []
        self.cur_ts_completed_jobs = []
        self.rejected_jobs = []
        self.not_ready_jobs = set()

        self.exit_flag = False

        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()
        self.scaling_overhead = 0
        self.testing_overhead = 0


    def _set_cluster_config(self):
        cluster_num_nodes = len(params.NODE_LIST)
        cpu_per_node = params.CPU_PER_NODE
        mem_per_node = params.MEM_PER_NODE
        #bw_per_node = params.BW_PER_NODE
        gpu_per_node = params.GPU_PER_NODE
        self.cluster_num_cpu = cluster_num_nodes * cpu_per_node
        self.cluster_num_mem = cluster_num_nodes * mem_per_node
        self.cluster_num_gpu = cluster_num_nodes * gpu_per_node
        self.cluster_num_bw = params.B
        self.cluster_B = params.B


    def set_exit_flag(self, exit_flag):
        self.exit_flag = exit_flag
        self.estimator.set_exit_flag(exit_flag)

    def _msg_handle(self):
        while not self.exit_flag:
            try:
                (t, src, dest, type, job) = self.scheduler_queue.get(False)
            except:
                continue
            self.logger.debug(self.name + ":: " + str((t, src, dest, type, job)))
            assert t == self.timer.get_clock()
            assert dest == "scheduler"

            if type == "submission" and src == "generator":
                if job is None:
                    # generator has finished the timeslot
                    self._schedule(t)
                else:
                    job.status = 'queueing'
                    # priority queue based on arrival time
                    self.queueing_jobs.put((job.arrival_time, job))
                    if job not in self.uncompleted_jobs:
                        self.uncompleted_jobs.append(job)
                    else:
                        raise RuntimeError
            elif type == "completion" and src == "progressor":
                if job is None:
                    # progressor has finished the timeslot
                    self._delete()
                else:
                    self.logger.info("#####################################################")
                    self.cur_ts_completed_jobs.append(job)
            elif type == "completion" and src == "statsor":
                if job is None:
                    # statsor finishes, start next timeslot
                    self._start_next_ts()
                else:
                    raise RuntimeError
        self.logger.debug(self.name + ":: " + self.name + " has exited.")

    def _price_update(self):
        for x in range(1, params.T + 1):
            self.price_B[x] = math.pow(params.base_B, self.used_b[x]/params.B) - 1
            for r in range(len(params.NODE_LIST)):
                self.price_cpu[x][r] = math.pow(params.base_res[0],
                                                self.node_used_cpu_list[x][r] / params.CPU_PER_NODE) - 1
                self.price_mem[x][r] = math.pow(params.base_res[1],
                                                self.node_used_mem_list[x][r] / params.MEM_PER_NODE) - 1
                self.price_gpu[x][r] = math.pow(params.base_res[2],
                                                self.node_used_gpu_list[x][r] / params.GPU_PER_NODE) - 1

    def _resource_update(self,job):
        for t in job.num_ps_t.keys():
            for r in range(len(params.NODE_LIST)):
                self.node_used_cpu_list[t][r] = self.node_used_cpu_list[t][r] + job.ps_placement_t[t][r] * job.ps_cpu + \
                                                job.worker_placement_t[t][r] * job.worker_cpu
                self.node_used_mem_list[t][r] = self.node_used_mem_list[t][r] + job.ps_placement_t[t][r] * job.ps_mem + \
                                                job.worker_placement_t[t][r] * job.worker_mem
                self.node_used_gpu_list[t][r] += job.worker_placement_t[t][r] * job.worker_gpu
            self.used_b[t] = self.used_b[t] + 2 * sum(map(sum,job.tran_data[t])) + \
                             job.params_size * (sum(job.worker_placement_t[t])-max(job.worker_placement_t[t]))


    def _schedule(self,time_slot):

        new_jobs = []
        while not self.queueing_jobs.empty():
            (arrival_time, job) = self.queueing_jobs.get()
            new_jobs.append(job)

        if time_slot > params.T: #delete jobs unfinished within params.T
            self.logger.debug(self.name + ":: " + "Starting to delete unfinished jobs!! ")
            self._delete_unfinished_jobs()

        test_tic = time.time()
        # first estimate speed
        self.estimator.existing_jobs = self.uncompleted_jobs + self.completed_jobs #
        self.logger.debug(self.name + ":: " + "newly arrived jobs: " + str(new_jobs))
        #self.estimator.test_speed(new_jobs)    #job.training_speeds[(1,1)]
        self.logger.debug("FINISH TESTING SPEED FOR NEW JOBS.")
        test_toc = time.time()
        self.testing_overhead += (test_toc - test_tic)


        tic = time.time()
        for job in new_jobs:
            #cost_i=float("inf")
            u_i = 0
            job.P = 2*random.randint(10,15) #skip testing speed phase
            self.logger.debug(self.name + "job.P: " + str(job.P))
            data_size = job.data_chunks

            self.logger.info(self.name + " :: " + "data sizes " + str(data_size))
            #self.logger.info("set_data_dist: " + str(job.train_data))
            #self.logger.info("set_data_dist: " + str(job.trained_data))

            #record dpc
            self.costi = {}
            #num
            self.ps={}
            self.worker={}
            #placement
            self.ps_p={}
            self.worker_p={}
            self.trained={}
            self.tran={}


            for t in range(job.arrival_slot,params.T+1):
                #schedule initialization
                self.num_ps={}
                self.num_worker={}
                self.ps_placement={}
                self.worker_placement={}
                self.trained_data={}
                self.tran_data={}

                cost =self.DPC(job,t,data_size)
                #self.logger.info("cost: " +str(cost))

                self.logger.info("Finish " +str(t) + "-th completion time testing for jobs!")

                self.costi[(t,data_size)]=cost
                self.ps[(t,data_size)] = copy.deepcopy(self.num_ps)
                self.worker[(t,data_size)] = copy.deepcopy(self.num_worker)
                self.ps_p[(t,data_size)]=copy.deepcopy(self.ps_placement)
                self.worker_p[(t,data_size)]=copy.deepcopy(self.worker_placement)
                self.trained[(t,data_size)]=copy.deepcopy(self.trained_data)
                self.tran[(t,data_size)]=copy.deepcopy(self.tran_data)

                '''
                utility function
                '''
                fi = 3000*random.uniform(1, 2) / (1 + math.exp((t - job.arrival_slot)/3))
                #self.logger.info("fi " +str(fi))
                #self.logger.info("u_i " +str(u_i))
                #self.logger.info("fi-cost " +str(fi-cost))
                if u_i < fi-cost:
                    #litter cost
                    u_i = fi-cost
                    
                    #Save scheduling information
                    job.cost = cost
                    job.social_welfare = fi
                    job.end_slot = t
                    job.num_ps_t = copy.deepcopy(self.num_ps)
                    job.num_worker_t = copy.deepcopy(self.num_worker)
                    job.ps_placement_t = copy.deepcopy(self.ps_placement)
                    job.worker_placement_t = copy.deepcopy(self.worker_placement)

                    job.trained_data = copy.deepcopy(self.trained_data)
                    job.tran_data = copy.deepcopy(self.tran_data)




            #self.logger.info("Final trained_data: " + str(job.trained_data) )
            #self.logger.info("Final tran_data: " + str(job.tran_data))
            
            '''
            sum_data_size =0
            for t in range(job.arrival_slot, job.arrival_slot + len(job.trained_data)):
                sum_data_size  += sum(job.trained_data[t])
            for t in range(job.arrival_slot-1, job.arrival_slot + len(job.tran_data)-1):
                for r in range(len(params.NODE_LIST)):
                    sum_data_size += sum(job.tran_data[t][r])
            self.logger.info("Final trained data size: " + str(sum_data_size))
            '''
            #Update resource usage and price after determining scheduling
            if u_i > 0 :
                self._resource_update(job)
                self._price_update()
                job.accept = 1
                self.social_welfare += job.social_welfare
                self.payoff += job.cost

        # check the scheduling result
        for job in self.uncompleted_jobs:
            if time_slot in job.num_ps_t.keys():
                job.num_ps=job.num_ps_t[time_slot]
            else:
                job.num_ps =0
            if time_slot in job.num_worker_t.keys():
                job.num_worker=job.num_worker_t[time_slot]
            else:
                job.num_worker =0

            self.logger.debug(self.name + ":: scheduling results" + " num_ps: " + str(job.num_ps) + " num_worker: " + \
                str(job.num_worker))
            self.logger.debug(self.name + " job.num_ps_t:" + str(job.num_ps_t))
            self.logger.debug(self.name + " job.num_worker_t:" + str(job.num_worker_t))

        scaling_tic = time.time()

        toc = time.time()

        self.logger.debug(self.name + " 277:: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")

        self.logger.info("###########" + self.name + " 377:: Total payoff of the service provider: " + str(self.payoff) + " ###########")


        self.logger.info("###########" + self.name + " 278:: Total social welfare: " + str(self.social_welfare) + " ###########")


        '''
        #the number of data that trained in next time slot
        for job in self.uncompleted_jobs:
            job.tran_data[time_slot+1][][]  # h_t^{rr'}
        '''

        self.running_jobs = []
        # send message to progress to update job progress
        thread_list = []
        for job in self.uncompleted_jobs:
            ps_placement=[]
            if time_slot in job.ps_placement_t.keys():
                self.logger.info("time_slot: " +str(time_slot))
                self.logger.info("job.PS_placement_t[time_slot]: " +str(job.ps_placement_t[time_slot]))
                for r in range(len(params.NODE_LIST)):
                    for i in range(job.ps_placement_t[time_slot][r]):
                        ps_placement.append(params.NODE_LIST[r])
            worker_placement=[]
            if time_slot in job.worker_placement_t.keys():
                self.logger.info("time_slot: " +str(time_slot))
                self.logger.info("job.worker_placement_t[time_slot]: " +str(job.worker_placement_t[time_slot]))
                for r in range(len(params.NODE_LIST)):
                    for i in range(job.worker_placement_t[time_slot][r]):
                        worker_placement.append(params.NODE_LIST[r])
            #self.logger.info("time_slot: "+ str(time_slot) + " ; job.num_ps_t.keys()[-1]: " + str(job.num_ps_t.keys()[-1]))


            if len(ps_placement) > 0 and len(worker_placement) > 0 : #and time_slot <= job.num_ps_t.keys()[-1]:

                # this may cause many ssh connections on a server and an error "ssh_exchange_identification: Connection closed by remote host"
                # to avoid this error, run 'echo "MaxStartups 100:10:200" | sudo tee -a /etc/ssh/sshd_config && sudo service ssh restart' on the server
                
                self.logger.debug(
                   self.name + ":: " + "job length of placement: " + str(len(ps_placement)) + ' '+ str(len(worker_placement)))
                self.running_jobs.append(job)
                '''
                thread = threading.Thread(target=self.__run, args=(job, ps_placement, worker_placement,time_slot,))
                thread.start()
                thread_list.append(thread)
                
                '''
                job.status = 'running'
                # send message to progressor
                msg = (self.timer.get_clock(), 'scheduler', 'progressor', 'running', job)
                self.hub_queue.put(msg)
            else:
                job.status = 'pending'

                # send message to progressor
                msg = (self.timer.get_clock(), 'scheduler', 'progressor', 'pending', job)
                self.hub_queue.put(msg)

        for thread in thread_list:
            thread.join()
        scaling_toc = time.time()
        self.scaling_overhead += (scaling_toc - scaling_tic)
        self.logger.debug(
            self.name + " 286:: " + "job starting time: " + "%.3f" % (scaling_toc - scaling_tic) + " seconds.")

        # send message to progressor to signal scheduling completion
        msg = (self.timer.get_clock(), 'scheduler', 'progressor', 'done', None)
        self.hub_queue.put(msg)


    def DPC(self,job,com_t,data):

        if data == 0: return 0

        min_cost = float("inf")
        if com_t == job.arrival_slot:
            #self.logger.info("Processing " + str(com_t) + "-th DPC")
            cost = self.DPC_T(job,com_t,data)
            return cost
        #Initial schedule and resource situation
        num_ps = copy.deepcopy(self.num_ps)
        num_worker = copy.deepcopy(self.num_worker)
        ps_placement = copy.deepcopy(self.ps_placement)
        worker_placement = copy.deepcopy(self.worker_placement)
        trained_data = copy.deepcopy(self.trained_data)
        tran_data = copy.deepcopy(self.tran_data)


        for d in range(data+1):
            #Initialization
            self.num_ps = copy.deepcopy(num_ps)
            self.num_worker = copy.deepcopy(num_worker)
            self.ps_placement = copy.deepcopy(ps_placement)
            self.worker_placement = copy.deepcopy(worker_placement)
            self.trained_data = copy.deepcopy(trained_data)
            self.tran_data = copy.deepcopy(tran_data)


            #Already calculated
            if (com_t-1, data-d) in self.costi:

                cost = self.costi[(com_t-1, data-d)]
                self.num_ps = copy.deepcopy(self.ps[(com_t-1, data-d)])
                self.num_worker = copy.deepcopy(self.worker[(com_t-1, data-d)])
                self.ps_placement=copy.deepcopy(self.ps_p[(com_t-1, data-d)])
                self.worker_placement=copy.deepcopy(self.worker_p[(com_t-1, data-d)])
                self.trained_data=copy.deepcopy(self.trained[(com_t-1, data-d)])
                self.tran_data=copy.deepcopy(self.tran[(com_t-1, data-d)])

            else:#calculate and save
                cost = self.DPC(job, com_t-1, data-d)

                self.costi[(com_t-1, data-d)]=cost
                self.ps[(com_t-1, data-d)] = copy.deepcopy(self.num_ps)
                self.worker[(com_t-1, data-d)]= copy.deepcopy(self.num_worker)
                self.ps_p[(com_t-1, data-d)]=copy.deepcopy(self.ps_placement)
                self.worker_p[(com_t-1, data-d)]=copy.deepcopy(self.worker_placement)
                self.trained[(com_t-1, data-d)]=copy.deepcopy(self.trained_data)
                self.tran[(com_t-1, data-d)]=copy.deepcopy(self.tran_data)

            cost_t = self.DPC_T(job, com_t, d)

            #self.logger.info("cost_t: " +str(cost_t))
            #self.logger.info("cost: " +str(cost))
            if min_cost > cost_t + cost:
                min_cost = cost_t + cost
                #save scheduling and resource usage temporarily
                r_num_ps=copy.deepcopy(self.num_ps)
                r_num_worker=copy.deepcopy(self.num_worker)
                r_ps_placement=copy.deepcopy(self.ps_placement)
                r_worker_placement=copy.deepcopy(self.worker_placement)
                r_trained_data=copy.deepcopy(self.trained_data)
                r_tran_data=copy.deepcopy(self.tran_data)
                #self.logger.info("r_num_ps: " +str(r_num_ps))
                #self.logger.info("r_num_worker: " +str(r_num_worker))

        #Optimal solution save and return
        if min_cost != float("inf"):
            self.num_ps = copy.deepcopy(r_num_ps)
            self.num_worker = copy.deepcopy(r_num_worker)
            self.ps_placement = copy.deepcopy(r_ps_placement)
            self.worker_placement = copy.deepcopy(r_worker_placement)
            self.trained_data = copy.deepcopy(r_trained_data)
            self.tran_data = copy.deepcopy(r_tran_data)

        #self.logger.info("self.num_ps: " + str(self.num_ps))
        #self.logger.info("self.num_worker: " + str(self.num_worker))
        #self.logger.info("self.ps_placement: " + str(self.ps_placement))
        #self.logger.info("self.worker_placement: " + str(self.worker_placement))
        #self.logger.info("min_cost " + str(min_cost))
        return min_cost

    def DPC_T(self, job, time, data):
        if data == 0:
            return 0
        # node number
        cluster_num_nodes = len(params.NODE_LIST)
        # Initialization
        self.num_ps[time] = 0
        self.num_worker[time] = 0
        self.ps_placement[time] = [0 for i in range(cluster_num_nodes)]
        self.worker_placement[time] = [0 for i in range(cluster_num_nodes)]
        self.trained_data[time] = [0 for i in range(cluster_num_nodes)]
        self.tran_data[time] = [[0 for i in range(cluster_num_nodes)] for i in range(cluster_num_nodes)]
        worker = [0 for i in range(cluster_num_nodes)]
        ps = [0 for i in range(cluster_num_nodes)]
        tran = [[0 for i in range(cluster_num_nodes)] for i in range(cluster_num_nodes)]
        trained = [0 for i in range(cluster_num_nodes)]

        M = [0 for i in range(cluster_num_nodes)]
        G = [0 for i in range(cluster_num_nodes)]
        cost_worker = {}  # [0 for i in range(cluster_num_nodes)]
        omega = {}  # [(r,r')]
        for r in range(cluster_num_nodes):
            M[r] = job.train_data[r]  # total training data
            for key in self.trained_data.keys():  # reduce trained data
                M[r] = M[r] - self.trained_data[key][r]
            for key in self.tran_data.keys():
                M[r] = M[r] - sum(self.tran_data[key][r])  # reduce tran data
            if job.worker_gpu == 0:
                G[r] = min(
                    max(params.CPU_PER_NODE - self.node_used_cpu_list[time][r] - job.ps_cpu, 0) // job.worker_cpu,
                    max(params.MEM_PER_NODE - self.node_used_mem_list[time][r] - job.ps_mem, 0) // job.worker_mem)
            else:
                G[r] = min(
                    max(params.CPU_PER_NODE - self.node_used_cpu_list[time][r] - job.ps_cpu, 0) // job.worker_cpu,
                    max(params.MEM_PER_NODE - self.node_used_mem_list[time][r] - job.ps_mem, 0) // job.worker_mem,
                    (params.GPU_PER_NODE - self.node_used_gpu_list[time][r]) // job.worker_gpu)
            # calculate price
            cost_worker[r] = job.worker_cpu * self.price_cpu[time][r] + job.worker_mem * self.price_mem[time][r] + \
                             job.worker_gpu * self.price_gpu[time][r]
            for r1 in range(cluster_num_nodes):
                if r == r1:
                    continue
                else:
                    # thet_i
                    omega[(r, r1)] = 2 * self.price_B[time] - (
                                job.worker_cpu * (self.price_cpu[time][r] - self.price_cpu[time][r1]) +
                                job.worker_mem * (self.price_mem[time][r] - self.price_mem[time][r1]) +
                                job.worker_gpu * (self.price_gpu[time][r] - self.price_gpu[time][
                            r1])) * job.num_epochs / job.P

                    # Sort by Q value in descending order
        R = sorted(omega.items(), key=lambda x: x[1])
        skip_site = []
        for j in range(len(R)):
            ((r, r1), value) = R[j]
            if value >= 0:
                break
            if (r1 in skip_site) or M[r] - sum(tran[r][:]) <= 0:
                continue

            # trained[r1] = min(M[r1], data - sum(trained) - sum(map(sum, tran)))
            tran[r][r1] = int(min(M[r] - sum(tran[r][:]), (params.B - self.used_b[time]) // 2,
                                  data - sum(trained) - sum(map(sum, tran)),
                                  G[r1] * (job.P / job.num_epochs) - M[r1] - sum(
                                      tran[i][r1] for i in range(cluster_num_nodes))))
            if tran[r][r1] <= 0:
                tran[r][r1] = 0
                skip_site.append(r1)

        order = sorted(cost_worker.items(), key=lambda x: x[1])
        for j in range(cluster_num_nodes):
            (r, value) = order[j]

            worker[r] = int(min(G[r], int(math.ceil(float(data * job.num_epochs) / job.P)) - sum(worker),
                                math.ceil((M[r] + sum(tran[i][r] for i in range(cluster_num_nodes))) * job.num_epochs / job.P)))

            #self.logger.debug(self.name + "490:: G[r] " + str(G[r]))
            #self.logger.debug(self.name + "490:: worker_2 " + str(int(math.ceil(float(data * job.num_epochs) / job.P)) - sum(worker)))
            #self.logger.debug(self.name + "490:: worker_3 " + str(math.ceil((M[r] + sum(tran[i][r] for i in range(cluster_num_nodes))) * job.num_epochs / job.P)))
            #self.logger.debug(self.name + "490:: worker[r] " + str(worker[r]))
            
            if worker[r] > 0:
                trained[r] = int(min(worker[r] * job.P / job.num_epochs - sum(tran[i][r] for i in range(cluster_num_nodes)),
                                 data - sum(trained) - sum(map(sum, tran))))
        # lack of worker or B
        #self.logger.debug(self.name + "490:: sum(worker) " + str(sum(worker)) )
        #self.logger.debug(self.name + "491:: math.ceil(float(data * job.num_epochs) / job.P) " + str(math.ceil(float(data * job.num_epochs) / job.P)) )
        #self.logger.debug(self.name + "492:: BBBB " + str((params.B - self.used_b[time] - 2 * sum(map(sum, tran)) - job.params_size * (sum(worker) - max(worker)))) )

        if sum(worker) < math.ceil(float(data * job.num_epochs) / job.P) or \
                (params.B - self.used_b[time] - 2 * sum(map(sum, tran)) - job.params_size * (
                        sum(worker) - max(worker))) < 0:
            cost = float('inf')
            #self.logger.info("############")
            return cost
        r_ps = worker.index(max(worker))
        ps[r_ps] = 1
        cost1 = 2 * sum(map(sum, tran)) * self.price_B[time] + job.params_size * (sum(worker) - max(worker)) * \
                self.price_B[time]
        # worker & PS in the same server?
        cost2 = 0
        for r in range(cluster_num_nodes):
            cost2 = cost2 + (worker[r] * job.worker_cpu + ps[r] * job.ps_cpu) * self.price_cpu[time][r] + \
                    (worker[r] * job.worker_mem + ps[r] * job.ps_mem) * self.price_mem[time][r] + \
                    (worker[r] * job.worker_gpu) * self.price_gpu[time][r]
        cost = cost1 + cost2
        self.num_ps[time] = 1
        self.num_worker[time] = sum(worker)
        self.ps_placement[time] = copy.deepcopy(ps)
        self.worker_placement[time] = copy.deepcopy(worker)
        self.trained_data[time] = copy.deepcopy(trained)
        self.tran_data[time] = copy.deepcopy(tran)
        #self.logger.debug(self.name + "525::  cost_t " + str(cost) )
        return cost



    def __run(self, job, ps_placement, worker_placement,time_slot):
        self.logger.debug(self.name + ":: " + job.name + ", num_ps: " + str(job.num_ps) + ", num_worker: " + str(
            job.num_worker) + ", ps placement: " + str(ps_placement) + ", worker placement: " + str(worker_placement))

        # set placement and start job
        # sys.exit()
        job.set_ps_placement(ps_placement)
        job.set_worker_placement(worker_placement)
        #job.data_upload(time_slot)
        job.start()

    def _delete(self):

        for job in self.cur_ts_completed_jobs:
            self.logger.debug("uncompleted_jobs: " + str(len(self.uncompleted_jobs)))
            if job in self.uncompleted_jobs:
                self.uncompleted_jobs.remove(job)

            self.logger.debug("uncompleted_jobs: " + str(len(self.uncompleted_jobs)))

            if job in self.running_jobs:
                self.running_jobs.remove(job)

            if job not in self.completed_jobs:
                self.completed_jobs.append(job)

        self.cur_ts_completed_jobs = []

        delete_tic = time.time()

        # clear existing jobs for next time slot
        #for job in self.running_jobs:
        #    job.delete(True)
        delete_toc = time.time()
        self.scaling_overhead += (delete_toc - delete_tic)
        self.logger.debug(self.name + ":: " + "job shutdown time: " + "%.3f" % (delete_toc - delete_tic) + " seconds.")

        # send message to statsor to get statistics of this timeslot
        msg = (self.timer.get_clock(), 'scheduler', 'statsor', 'control', None)
        self.hub_queue.put(msg)

    def _delete_unfinished_jobs(self):
        for job in self.uncompleted_jobs:
            self.uncompleted_jobs.remove(job)
            #self.running_jobs.remove(job)
            self.rejected_jobs.append(job)

        # clear existing jobs for next time slot
        #for job in self.rejected_jobs:
        #    job.delete(True)


    def _start_next_ts(self):
        # send message to timer to signal starting next timeslot
        msg = (self.timer.get_clock(), 'scheduler', 'timer', 'control', None)
        self.hub_queue.put(msg)
