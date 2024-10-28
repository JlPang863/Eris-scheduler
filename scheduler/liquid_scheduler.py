from __future__ import division
import random
import copy
import Queue
import os
import time
import sys
import threading
import math
import params
from estimator import Estimator


class Liquid_Scheduler(object):
    def __init__(self, name, logger, scheduler_queue, hub_queue, timer):
        self.name = name  # e.g., 'UTIL'
        self.logger = logger
        self.scheduler_queue = scheduler_queue
        self.hub_queue = hub_queue
        self.timer = timer

        # self.cluster_num_cpu = None
        # self.cluster_num_mem = None
        # self.cluster_num_gpu = None

        self.cluster_used_cpu = 0
        self.cluster_used_mem = 0
        self.cluster_used_gpu = 0
        self.cluster_used_bw = 0
        self._set_cluster_config()

        cluster_num_nodes = len(params.NODE_LIST)
        cpu_per_node = params.CPU_PER_NODE
        mem_per_node = params.MEM_PER_NODE
        bw_per_node = params.BW_PER_NODE
        gpu_per_node = params.GPU_PER_NODE
        self.cluster_num_cpu = cluster_num_nodes * cpu_per_node
        self.cluster_num_mem = cluster_num_nodes * mem_per_node
        self.cluster_num_gpu = cluster_num_nodes * gpu_per_node
        self.cluster_num_bw = cluster_num_nodes * bw_per_node



        #resource usage
        self.node_used_cpu_list = {}   #[time][r]
        self.node_used_mem_list = {}
        self.node_used_gpu_list = {}
        self.used_b = {}     #[time][r][r']

        # resource price
        self.drf = 0.1
        self.price_B = {}  # [time]
        self.price_cpu = {}  # [time][r]
        self.price_mem = {}
        self.price_gpu = {}

        for x in range(1, params.T + 1):#10 extra time slot for overflow
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
        self.not_ready_jobs = set()
        self.rejected_jobs = []

        self.exit_flag = False

        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()
        self.scaling_overhead = 0
        self.testing_overhead = 0

    def _set_cluster_config(self):
        cluster_num_nodes = len(params.NODE_LIST)
        cpu_per_node = params.CPU_PER_NODE
        mem_per_node = params.MEM_PER_NODE
        bw_per_node = params.BW_PER_NODE
        gpu_per_node = params.GPU_PER_NODE
        self.cluster_num_cpu = cluster_num_nodes * cpu_per_node
        self.cluster_num_mem = cluster_num_nodes * mem_per_node
        self.cluster_num_gpu = cluster_num_nodes * gpu_per_node
        self.cluster_num_bw = cluster_num_nodes * bw_per_node

        #self.cluster_num_bw =  bw_per_node
        #self.cluster_cost_bw = bw_per_node_cost

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

            self.used_b[t] = self.used_b[t] + job.params_size * (sum(job.worker_placement_t[t])-max(job.worker_placement_t[t]))
        for t in job.tran_data.keys():
            tran_per_t= 2 * sum(map(sum, job.tran_data[t])) / (job.end_slot - job.arrival_slot + 1)
            for x in range(job.arrival_slot, job.end_slot + 1):
                self.used_b[x] += tran_per_t


    
    def _schedule(self,time_slot):

        cluster_num_nodes = len(params.NODE_LIST)

        new_jobs = []
        while not self.queueing_jobs.empty(): #schedule new job
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


            job.P = random.randint(5,10) #skip testing speed phase

            '''
            design the score metric for scheduling
            '''
            
            dom_share = max(float(job.worker_cpu) / self.cluster_num_cpu, float(job.worker_mem )/ self.cluster_num_mem,
                            float(job.worker_gpu) / self.cluster_num_gpu)

            # worker_num denote the maximum number of workers 
            worker_num = int(math.ceil(float(self.drf) / dom_share))

            self.logger.debug(self.name + "152:: " + "random worker-num: " + str(worker_num))

            cost_of_worker = [0 for i in range(cluster_num_nodes)]

            #calculate the current cost for one worker
            for r in range(cluster_num_nodes): 
                cost_of_worker[r] = job.worker_cpu * self.price_cpu[job.arrival_slot][r] + job.worker_mem * self.price_mem[job.arrival_slot][r] + \
                             job.worker_gpu * self.price_gpu[job.arrival_slot][r]
            
            
            base_price_worker = 1
            cost_worker = max(base_price_worker, max(cost_of_worker))
            min_score = float("inf")
            min_worker_j = 0
            lambda_value = 0.5
            total_gpu_j = self.cluster_num_gpu
            used_gpu_j = sum(self.node_used_gpu_list[job.arrival_slot])       
            self.logger.debug(self.name + "152:: " + "used_gpu: " + str(used_gpu_j))

            for worker_j in xrange(worker_num):
                req_gpu_j = job.worker_gpu * worker_j
                fitness_j =   -1 * (req_gpu_j + used_gpu_j) / total_gpu_j
                self.logger.debug(self.name + "152:: " + "fitness_j: " + str(fitness_j))
                self.logger.debug(self.name + "152:: " + "cost_worker: " + str(cost_worker))

                score_j =  + (1-lambda_value) * fitness_j
                self.logger.debug(self.name + "152:: " + "cost: " + str(lambda_value * cost_worker * worker_j))
                self.logger.debug(self.name + "152:: " + "fitness_j: " + str(fitness_j))

                if score_j < min_score:
                    min_score = score_j
                    min_worker_j = worker_j
                    

            # the selected number of workers
            worker_num = min_worker_j 

            #self.logger.debug(self.name + "152:: " + "math.ceil(float(job.data_chunks * job.num_epochs)/job.P): " + str(math.ceil(float(job.data_chunks * job.num_epochs)/job.P)))
            self.logger.debug(self.name + "152:: " + "random worker-num: " + str(worker_num))
            job.run_time=int(math.ceil(float(job.data_chunks * job.num_epochs)/worker_num/job.P))
            self.logger.debug(self.name + "152:: " + "run-time: " + str(job.run_time))

            u_i=0

            for tsf in range(job.arrival_slot, params.T+2-job.run_time):
                self.num_ps = {}
                self.num_worker = {}
                self.ps_placement = {}
                self.worker_placement = {}
                self.trained_data = {}
                self.tran_data = {}

                ps = 0
                tef = tsf+job.run_time

                # place ps
                cost_ps = [0 for i in range(cluster_num_nodes)]
                for r in range(cluster_num_nodes):
                    cost_ps[r] = job.ps_cpu * self.price_cpu[tsf][r] + job.ps_mem * self.price_mem[tsf][r]

                ps = cost_ps.index(min(cost_ps))
                while cost_ps[ps] != float("inf"):
                    flag = 1
                    for t in range(tsf, tef):
                        if self.node_used_cpu_list[t][ps] + job.ps_cpu > params.CPU_PER_NODE or \
                                self.node_used_mem_list[t][ps] + job.ps_mem > params.MEM_PER_NODE:
                            flag = 0
                            break
                    if flag == 1:
                        break
                    else:
                        cost_ps[ps] = float("inf")
                        ps = cost_ps.index(min(cost_ps))
                if cost_ps[ps] == float("inf"): continue

                #worker placement
                flag, cost = self._placement_worker(job,tsf,tef,worker_num,ps)

                #
                if flag==0:  continue
                else:
                    fi = 3000 / (1 + math.exp((tef - 1 - job.arrival_slot)/3))
                    if fi - cost > 0:
                        u_i = fi - cost
                        job.social_welfare = fi
                        job.num_ps_t = copy.deepcopy(self.num_ps)
                        job.num_worker_t = copy.deepcopy(self.num_worker)
                        job.ps_placement_t = copy.deepcopy(self.ps_placement)
                        job.worker_placement_t = copy.deepcopy(self.worker_placement)
                        job.tran_data = copy.deepcopy(self.tran_data)
                        job.end_slot = tef - 1
                        break
                    else:
                        continue

            # Update resource usage and price after determining scheduling
            if u_i > 0:
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
                job.num_ps=0
            if time_slot in job.num_worker_t.keys():
                job.num_worker=job.num_worker_t[time_slot]
            else:
                job.num_worker=0
            self.logger.debug(self.name + ":: scheduling results" + " num_ps: " + str(job.num_ps) + " num_worker: " + \
                str(job.num_worker))
            self.logger.debug(self.name + " 266:: " + "job.worker_placement_t: " + str(job.num_worker_t))
            self.logger.debug(self.name + " 267:: " + "job.ps_placement_t: " + str(job.num_ps_t))
        scaling_tic = time.time()

        toc = time.time()
        self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")

        self.logger.info("###########" + self.name + " 377:: Total payoff of the service provider: " + str(self.payoff) + " ###########")

        self.logger.info("###########" + self.name + " 272:: Total social welfare: " + str(self.social_welfare) + " ###########")


        '''
        #the number of data that trained in next time slot
        for job in self.uncompleted_jobs:
            job.tran_data[time_slot+1][][]  # ---->h_t^{rr'}
        '''

        self.running_jobs = []
        # send message to progress to update job progress
        thread_list = []
        for job in self.uncompleted_jobs:
            ps_placement=[]
            if time_slot in job.ps_placement_t.keys():
                for r in range(len(params.NODE_LIST)):
                    for i in range(job.ps_placement_t[time_slot][r]):
                        ps_placement.append(params.NODE_LIST[r])
            worker_placement=[]
            if time_slot in job.worker_placement_t.keys():
                self.logger.debug(self.name + " 225:: " + "job.worker_placement_t: " + str(job.worker_placement_t[time_slot]))
                for r in range(len(params.NODE_LIST)):
                    for i in range(job.worker_placement_t[time_slot][r]):
                        worker_placement.append(params.NODE_LIST[r])
            self.logger.debug(
                    self.name + ":: " + "job length of placement: " + str(len(ps_placement)) + ' '+ str(len(worker_placement)))

            if len(ps_placement) > 0 and len(worker_placement) > 0:

                # this may cause many ssh connections on a server and an error "ssh_exchange_identification: Connection closed by remote host"
                # to avoid this error, run 'echo "MaxStartups 100:10:200" | sudo tee -a /etc/ssh/sshd_config && sudo service ssh restart' on the server

                '''
                self.running_jobs.append(job)
                thread = threading.Thread(target=self.__run, args=(job, ps_placement, worker_placement,))
                thread.start()
                thread_list.append(thread)
                job.status = 'running'
                '''

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
            self.name + ":: " + "job starting time: " + "%.3f" % (scaling_toc - scaling_tic) + " seconds.")

        # send message to progressor to signal scheduling completion
        msg = (self.timer.get_clock(), 'scheduler', 'progressor', 'done', None)
        self.hub_queue.put(msg)
    

    #placement
    def _placement_worker(self, job, tsf, tef, Nhated, ps):

        cluster_num_nodes = len(params.NODE_LIST)
        cost1 = 0

        cost_worker = {}

        for r in range(cluster_num_nodes):
            # calculate price
            cost_worker[r] = job.worker_cpu * self.price_cpu[tsf][r] + job.worker_mem * self.price_mem[tsf][r] + \
                             job.worker_gpu * self.price_gpu[tsf][r]

        # Sort by price value in non-descending order
        R = sorted(cost_worker.items(), key=lambda x: x[1])

        worker = [0 for i in range(len(params.NODE_LIST))]
        for j in range(len(R)):
            (site, value) = R[j]
            worker_t = [0 for i in range(tsf, tef)]
            for t in range(tsf, tef):
                used_cpu = self.node_used_cpu_list[t][site]
                used_mem = self.node_used_mem_list[t][site]
                used_gpu = self.node_used_gpu_list[t][site]
                if ps == site:
                    used_cpu += job.ps_cpu
                    used_mem += job.ps_mem
                if t == job.arrival_slot:
                    if job.worker_gpu == 0:
                        worker_t[t - tsf] = int(min((params.CPU_PER_NODE - used_cpu) // job.worker_cpu,
                                                    (params.MEM_PER_NODE - used_mem) // job.worker_mem,
                                                    int(math.ceil(
                                                        float(job.train_data[site] * job.num_epochs) / job.P))))
                    else:
                        worker_t[t - tsf] = int(min((params.CPU_PER_NODE - used_cpu) // job.worker_cpu,
                                                    (params.MEM_PER_NODE - used_mem) // job.worker_mem,
                                                    (params.GPU_PER_NODE - used_gpu) // job.worker_gpu, int(
                                math.ceil(float(job.train_data[site] * job.num_epochs) / job.P))))
                else:
                    if job.worker_gpu == 0:
                        worker_t[t - tsf] = int(min((params.CPU_PER_NODE - used_cpu) // job.worker_cpu,
                                                    (params.MEM_PER_NODE - used_mem) // job.worker_mem))
                    else:
                        worker_t[t - tsf] = int(min((params.CPU_PER_NODE - used_cpu) // job.worker_cpu,
                                                    (params.MEM_PER_NODE - used_mem) // job.worker_mem,
                                                    (params.GPU_PER_NODE - used_gpu) // job.worker_gpu))
                                                    
            self.logger.debug(self.name + ":: min(worker_t): " + str(min(worker_t)) )
            self.logger.debug(self.name + ":: Nhated: " + str(Nhated) )
            self.logger.debug(self.name + ":: worker[site]: " + str(worker[site]) )
            worker[site] = int(min(min(worker_t), Nhated))

            Nhated -= worker[site]
            # Effective scheduling, computing data transmission
            if Nhated == 0:

                train_data = [0 for i in range(cluster_num_nodes)]
                for r in range(cluster_num_nodes):
                    train_data[r] = worker[r] * job.P * job.run_time / job.num_epochs

                # data transmission

                tran = [[0 for i in range(cluster_num_nodes)] for i in range(cluster_num_nodes)]
                A = [0 for i in range(cluster_num_nodes)]
                for r in range(cluster_num_nodes):
                    A[r] = job.train_data[r] - train_data[r]

                total = sum(A)
                reduce = [0 for i in range(cluster_num_nodes)]
                extro = 0
                if total < 0:
                    extro = (-total) * job.num_epochs // job.P
                while extro != 0:
                    h = A.index(min(A))
                    reduce[h] = int(min(extro, worker[h]))
                    extro -= reduce[h]
                    A[h] += reduce[h] * job.P / job.num_epochs

                for r in range(cluster_num_nodes):
                    if A[r] < 0:
                        sum_rec = 0
                        R = sorted(cost_worker.items(), key=lambda x: x[1], reverse=True)
                        for j in range(len(R)):
                            (site, value) = R[j]
                            if site == r: continue
                            if A[site] > 0:
                                tran[site][r] = int(min(A[site] - sum(tran[site]), -A[r] - sum_rec))
                                sum_rec += tran[site][r]

                tran_per_t = int(math.ceil(float(2 * sum(map(sum, tran))) / (tef - job.arrival_slot)))
                for t in range(tsf, tef):
                    if (params.B - self.used_b[t] - tran_per_t - job.params_size * (sum(worker) - worker[ps])) < 0:
                        return 0, 0

                flag = 1
                ps_t = [0 for i in range(cluster_num_nodes)]
                ps_t[ps] = 1
                cost1 = 0
                cost2 = 0
                for t in range(tsf, tef - 1):
                    self.num_ps[t] = 1
                    self.num_worker[t] = sum(worker)
                    self.ps_placement[t] = copy.deepcopy(ps_t)
                    self.worker_placement[t] = copy.deepcopy(worker)

                    cost1 = cost1 + tran_per_t * self.price_B[t] + job.params_size * (sum(worker) - worker[ps]) * self.price_B[
                        t]
                    for r in range(cluster_num_nodes):
                        cost2 = cost2 + (worker[r] * job.worker_cpu + ps_t[r] * job.ps_cpu) * self.price_cpu[t][r] + \
                                (worker[r] * job.worker_mem + ps_t[r] * job.ps_mem) * self.price_mem[t][r] + \
                                (worker[r] * job.worker_gpu) * self.price_gpu[t][r]
                # tef-1
                for r in range(cluster_num_nodes):
                    worker[r] -= reduce[r]
                    cost2 = cost2 + (worker[r] * job.worker_cpu + ps_t[r] * job.ps_cpu) * self.price_cpu[tef - 1][r] + \
                            (worker[r] * job.worker_mem + ps_t[r] * job.ps_mem) * self.price_mem[tef - 1][r] + \
                            (worker[r] * job.worker_gpu) * self.price_gpu[tef - 1][r]
                cost1 = cost1 + tran_per_t * self.price_B[tef - 1] + job.params_size * (sum(worker) - worker[ps]) * \
                        self.price_B[tef - 1]
                self.num_ps[tef - 1] = 1
                self.logger.debug(self.name + ":: sum(worker)" + str(sum(worker)) )
                self.num_worker[tef - 1] = sum(worker)
                self.ps_placement[tef - 1] = copy.deepcopy(ps_t)
                self.worker_placement[tef - 1] = copy.deepcopy(worker)

                self.tran_data[tsf] = copy.deepcopy(tran)
                cost = cost1 + cost2

                return flag, cost

        return 0, 0

    def __run(self, job, ps_placement, worker_placement):
        self.logger.debug(self.name + ":: " + job.name + ", num_ps: " + str(job.num_ps) + ", num_worker: " + str(
            job.num_worker) + ", ps placement: " + str(ps_placement) + ", worker placement: " + str(worker_placement))

        # set placement and start job
        # sys.exit()
        job.set_ps_placement(ps_placement)
        job.set_worker_placement(worker_placement)
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
        for job in self.running_jobs:
            job.delete(True)
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
