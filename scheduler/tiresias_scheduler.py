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


class Tiresias_Scheduler(object):
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
        #self._set_cluster_config()

        cluster_num_nodes = len(params.NODE_LIST)
        cpu_per_node = params.CPU_PER_NODE
        mem_per_node = params.MEM_PER_NODE
        bw_per_node = params.BW_PER_NODE
        gpu_per_node = params.GPU_PER_NODE
        self.cluster_num_cpu = cluster_num_nodes * cpu_per_node
        self.cluster_num_mem = cluster_num_nodes * mem_per_node
        self.cluster_num_gpu = cluster_num_nodes * gpu_per_node
        self.cluster_num_bw = cluster_num_nodes * bw_per_node


        # resource usage
        self.node_used_cpu_list = {}  # [time][r]
        self.node_used_mem_list = {}
        self.node_used_gpu_list = {}
        self.used_b = {}  # [time]

        # resource price
        self.price_B = {}  # [time]
        self.price_cpu = {}  # [time][r]
        self.price_mem = {}
        self.price_gpu = {}

        for x in range(1, params.T + 1):
            self.node_used_cpu_list[x] = [0 for i in range(len(params.NODE_LIST))]
            self.node_used_mem_list[x] = [0 for i in range(len(params.NODE_LIST))]
            self.node_used_gpu_list[x] = [0 for i in range(len(params.NODE_LIST))]
            self.used_b[x] = 0
            '''
            self.price_cpu[x] = [0 for i in range(len(params.NODE_LIST))]
            self.price_mem[x] = [0 for i in range(len(params.NODE_LIST))]
            self.price_gpu[x] = [0 for i in range(len(params.NODE_LIST))]
            '''
            self.price_cpu[x] = {}
            self.price_mem[x] = {}
            self.price_gpu[x] = {}
            self.price_B[x] = {}
            self._price_update(x)

        # cluster=Cluster()
        # self.cluster=cluster
        self.social_welfare = 0
        self.payoff  = 0
        self.preemption_uency = 0
        self.uncompleted_job_arr = {}  # uncompleted jobs before time slot t,job-priority, 1-high,2-low
        self.running_job_arr = []  # running jobs before time slot t
        self.new_running_job = []

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
        bw_per_node = params.BW_PER_NODE
        gpu_per_node = params.GPU_PER_NODE
        self.cluster_num_cpu = cluster_num_nodes * cpu_per_node
        self.cluster_num_mem = cluster_num_nodes * mem_per_node
        self.cluster_num_gpu = cluster_num_nodes * gpu_per_node
        self.cluster_num_bw = cluster_num_nodes * bw_per_node

        # self.cluster_num_bw =  bw_per_node
        # self.cluster_cost_bw = bw_per_node_cost

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

    def _price_update(self, t):
        self.price_B[t] = math.pow(params.base_B, self.used_b[t]/params.B) - 1
        for r in range(len(params.NODE_LIST)):
            self.price_cpu[t][r] = math.pow(params.base_res[0],
                                            self.node_used_cpu_list[t][r] / params.CPU_PER_NODE) - 1
            self.price_mem[t][r] = math.pow(params.base_res[1],
                                            self.node_used_mem_list[t][r] / params.MEM_PER_NODE) - 1
            self.price_gpu[t][r] = math.pow(params.base_res[2],
                                            self.node_used_gpu_list[t][r] / params.GPU_PER_NODE) - 1

    def _resource_update(self,job,t):
        for r in range(len(params.NODE_LIST)):
            self.node_used_cpu_list[t][r] = self.node_used_cpu_list[t][r] + job.ps_placement_t[t][r] * job.ps_cpu + \
                                            job.worker_placement_t[t][r] * job.worker_cpu
            self.node_used_mem_list[t][r] = self.node_used_mem_list[t][r] + job.ps_placement_t[t][r] * job.ps_mem + \
                                            job.worker_placement_t[t][r] * job.worker_mem
            self.node_used_gpu_list[t][r] += job.worker_placement_t[t][r] * job.worker_gpu
        self.used_b[t] = self.used_b[t] + 2 * sum(map(sum,job.tran_data[t])) + \
                         job.params_size * (sum(job.worker_placement_t[t])-max(job.worker_placement_t[t]))
    
    def _recompute_priority(self):
        min_priority = float('inf')
        for job in self.uncompleted_job_arr.keys():
            job.j_priority = -job.accum_running_time * job.allocated_workers
            if job.j_priority < min_priority:
                min_priority = job.j_priority
        for job in self.uncompleted_job_arr:
            # if job.j_priority < min_priority/2:  # low priority
            #     job.discrete_priority = 2
            # else:   # high priority
            #     job.discrete_priority = 1
            if min_priority == 0 or job.j_priority == 0:
                discrete_priority = 1
            else:
                discrete_priority = math.ceil(float(job.j_priority) / min_priority / 0.5)
            self.uncompleted_job_arr[job] = discrete_priority

    def _schedule(self, time_slot):

        cluster_num_nodes = len(params.NODE_LIST)

        new_jobs = []
        while not self.queueing_jobs.empty():  # schedule new job
            (arrival_time, job) = self.queueing_jobs.get()
            new_jobs.append(job)

        if time_slot > params.T: #delete jobs unfinished within params.T
            self._delete_unfinished_jobs()

        test_tic = time.time()
        # first estimate speed

        self.estimator.existing_jobs = self.uncompleted_jobs + self.completed_jobs  #
        self.logger.debug(self.name + ":: " + "newly arrived jobs: " + str(new_jobs))
        #self.estimator.test_speed(new_jobs)    #job.training_speeds[(1,1)]
        self.logger.debug("FINISH TESTING SPEED FOR NEW JOBS.")
        test_toc = time.time()
        self.testing_overhead += (test_toc - test_tic)

        tic = time.time()

        self._recompute_priority()
        for job in new_jobs:
            #Random number of workers(evry)
            job.P = 10 #skip testing speed phase
            self.logger.debug(self.name + "152:: " + "self.node_used_cpu_list[time_slot]: " + str(self.node_used_cpu_list[time_slot]))
            if job.worker_gpu == 0:
                worker_upper_bound = int(min(
                    math.floor(len(params.NODE_LIST)*params.CPU_PER_NODE - sum(self.node_used_cpu_list[time_slot])/job.worker_cpu) ,
                    math.floor(len(params.NODE_LIST)*params.MEM_PER_NODE - sum(self.node_used_mem_list[time_slot])/job.worker_mem)))
            else:
                worker_upper_bound = int(min(
                    math.floor(len(params.NODE_LIST)*params.CPU_PER_NODE - sum(self.node_used_cpu_list[time_slot])/job.worker_cpu)  ,
                    math.floor(len(params.NODE_LIST)*params.MEM_PER_NODE - sum(self.node_used_mem_list[time_slot])/job.worker_mem) ,
                    math.floor(len(params.NODE_LIST)*params.GPU_PER_NODE - sum(self.node_used_gpu_list[time_slot])/job.worker_gpu) ))

            #worker_num=random.randint(math.floor(0.3*min(worker_upper_bound, int(math.ceil(job.data_chunks/job.P)))),math.floor(0.6*min(worker_upper_bound, int(math.ceil(job.data_chunks/job.P)))))  # I
            job.worker_num = random.randint(2, 2+math.floor(min(worker_upper_bound, int(math.ceil(float(job.data_chunks * job.num_epochs)/job.P)))))
            self.logger.debug(self.name + "152:: " + "worker_upper_bound: " + str(worker_upper_bound))
            self.logger.debug(self.name + "152:: " + "math.ceil(job.num_epochs * job.data_chunks/job.P): " + str(math.ceil(float(job.data_chunks * job.num_epochs)/job.P)))
            self.logger.debug(self.name + "152:: " + " worker-num: " + str(job.worker_num))
            job.total_worker = int(math.ceil(float(job.data_chunks * job.num_epochs)/job.P))
            job.allocated_workers = 0
            job.accum_running_time = 0
            job.cost = 0
            #self.logger.debug(self.name + "152:: " + "run-time: " + str(job.run_time))
            
            self.uncompleted_job_arr[job] = 1

        # sort according to the priority in non-decreasing
        for tuple in sorted(self.uncompleted_job_arr.items(), key=lambda x: x[1], reverse=False):
            job = tuple[0]
            # job.worker_placement = []
            #j_worker_type = job.w_type
            worker = [0 for i in range(cluster_num_nodes)]
            ps_t = [0 for i in range(cluster_num_nodes)]
            trained = [0 for i in range(cluster_num_nodes)]
            tran = [[0 for i in range(cluster_num_nodes)] for i in range(cluster_num_nodes)]

            
            # ps placement
            cost_ps = [0 for i in range(cluster_num_nodes)]
            cost_worker = {}
            for r in range(cluster_num_nodes):
                #self.logger.debug(self.name + ":: price_cpu: " + str(self.price_cpu[time_slot][r]))
                #self.logger.debug(self.name + ":: price_mem: " + str(self.price_mem[time_slot][r]))
                cost_ps[r] = job.ps_cpu * self.price_cpu[time_slot][r] + job.ps_mem * self.price_mem[time_slot][r]
                cost_worker[r] = job.worker_cpu * self.price_cpu[time_slot][r] + job.worker_mem * self.price_mem[time_slot][r] + \
                                 job.worker_gpu * self.price_gpu[time_slot][r]

            ps = cost_ps.index(min(cost_ps))
            while cost_ps[ps] != float("inf"):
                if self.node_used_cpu_list[time_slot][ps] + job.ps_cpu > params.CPU_PER_NODE or \
                        self.node_used_mem_list[time_slot][ps] + job.ps_mem > params.MEM_PER_NODE:
                    cost_ps[ps] = float("inf")
                    ps = cost_ps.index(min(cost_ps))
                else:
                    break
            #resources are not enough to place a PS
            if cost_ps[ps] == float("inf"):
                continue

            #worker placement
            G = [0 for i in range(cluster_num_nodes)]
            R = sorted(cost_worker.items(), key=lambda x: x[1])
            worker_num_needed = job.worker_num
            for j in range(cluster_num_nodes):
                (r, value) = R[j]
                used_cpu = self.node_used_cpu_list[time_slot][r]
                used_mem = self.node_used_mem_list[time_slot][r]
                used_gpu = self.node_used_gpu_list[time_slot][r]
                if (r == ps):
                    used_cpu += job.ps_cpu
                    used_mem += job.ps_mem
                    # used_gpu += job.ps_gpu
                if job.worker_gpu == 0:
                    G[r] = int(min((params.CPU_PER_NODE - used_cpu) // job.worker_cpu,
                                   (params.MEM_PER_NODE - used_mem) // job.worker_mem))
                else:
                    G[r] = int(min((params.CPU_PER_NODE - used_cpu) // job.worker_cpu,
                                   (params.MEM_PER_NODE - used_mem) // job.worker_mem,
                                   (params.GPU_PER_NODE - used_gpu) // job.worker_gpu))

                worker[r] = int(min(G[r], job.total_worker - job.allocated_workers - sum(worker), worker_num_needed - sum(worker)))

            # calculate A
            A = [0 for i in range(cluster_num_nodes)]
            # calculate M
            M = [0 for i in range(cluster_num_nodes)]

            # resources are not enough to place workers
            if sum(worker) < min(math.ceil(float(sum(M))*job.num_epochs/job.P), worker_num_needed):
                continue

            # data transmission
            train_data = [0 for i in range(cluster_num_nodes)]


            for r in range(cluster_num_nodes):
                M[r] = job.train_data[r]  # total training data
                for key in job.trained_data.keys():  # reduce trained data
                    M[r] = M[r] - job.trained_data[key][r]
                for key in job.tran_data.keys():
                    M[r] = M[r] - sum(job.tran_data[key][r])  # reduce tran data

            for r in range(cluster_num_nodes):
                # calculate training data according to worker placement
                train_data[r] = int(min(worker[r] * job.P / job.num_epochs, sum(M) - sum(train_data)))
                A[r] = M[r] - train_data[r]
            for r in range(cluster_num_nodes):
                if A[r] < 0:
                    trained[r] = M[r]
                    sum_rec = 0
                    qcost = {}
                    R = sorted(cost_worker.items(), key=lambda x: x[1], reverse=True)
                    for j in range(len(R)):
                        (site, value) = R[j]
                        if site == r: continue
                        if A[site] > 0:
                            tran[site][r] = min(A[site] - sum(tran[site]), -A[r] - sum_rec)
                            sum_rec += tran[site][r]
                else:
                    trained[r] = train_data[r]
            # B is not enough
            if params.B - self.used_b[time_slot] - 2 * sum(map(sum, tran)) - job.params_size * (sum(worker) - worker[ps]) < 0:
                continue
            else:
                ps_t[ps] = 1
                job.num_ps_t[time_slot] = 1
                job.num_worker_t[time_slot] = sum(worker)
                job.ps_placement_t[time_slot] = copy.deepcopy(ps_t)
                job.worker_placement_t[time_slot] = copy.deepcopy(worker)
                job.trained_data[time_slot] = copy.deepcopy(trained)
                job.tran_data[time_slot] = copy.deepcopy(tran)
                job.accum_running_time += 1
                job.allocated_workers += job.worker_num

                job.cost = job.cost + 2 * sum(map(sum, tran)) * self.price_B[time_slot] + \
                           job.params_size * (sum(worker) - worker[ps]) * self.price_B[time_slot]
                for r in range(cluster_num_nodes):
                    job.cost = job.cost + (worker[r] * job.worker_cpu + ps_t[r] * job.ps_cpu) * self.price_cpu[time_slot][r] + \
                            (worker[r] * job.worker_mem + ps_t[r] * job.ps_mem) * self.price_mem[time_slot][r] + \
                            (worker[r] * job.worker_gpu) * self.price_gpu[time_slot][r]

                # update resource and price
                self._resource_update(job, time_slot)
                self._price_update(time_slot)

                self.new_running_job.append(job)

        for job in self.new_running_job:
            self.running_job_arr.append(job)
        #self.running_job_arr = copy.deepcopy(self.new_running_job)
        self.new_running_job = []

        
        # judge whether the job is complete after this slot
        complete = []
        for job in self.uncompleted_job_arr.keys():
            if job.allocated_workers == job.total_worker:
                complete.append(job)
                job.slots_completed = time_slot
                fi = 3000 / (1 + math.exp((time_slot - job.arrival_slot) / 3))
                if fi - job.cost > 0:
                    job.social_welfare = fi
                else:
                    self.uncompleted_jobs.remove(job)
                    self.rejected_jobs.append(job)
                self.social_welfare += job.social_welfare
                self.payoff = job.cost

        for job in complete:
            del self.uncompleted_job_arr[job]
    

        # check the scheduling result
        for job in self.uncompleted_jobs:
            if time_slot in job.num_ps_t.keys():
                job.num_ps = job.num_ps_t[time_slot]
            else:
                job.num_ps = 0
            if time_slot in job.num_worker_t.keys():
                job.num_worker = job.num_worker_t[time_slot]
            else:
                job.num_worker = 0

            self.logger.debug(self.name + ":: scheduling results" + " num_ps: " + str(job.num_ps) + " num_worker: " + \
                              str(job.num_worker))
            self.logger.debug(self.name + " job.num_ps_t:" + str(job.num_ps_t))
            self.logger.debug(self.name + " job.num_worker_t:" + str(job.num_worker_t))
            #job.get_remain_workload(time_slot)

        scaling_tic = time.time()

        toc = time.time()
        self.logger.debug(self.name + " 375:: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")

        
        self.logger.info("###########" + self.name + " 377:: Total payoff of the service provider: " + str(self.payoff) + " ###########")

        self.logger.info("###########" + self.name + " 377:: Total social welfare: " + str(self.social_welfare) + " ###########")



        self.running_jobs = []
        # send message to progress to update job progress
        thread_list = []
        for job in self.uncompleted_jobs:
            ps_placement = []
            if time_slot in job.ps_placement_t.keys():
                self.logger.info("time_slot: " + str(time_slot))
                self.logger.info("job.PS_placement_t[time_slot]: " + str(job.ps_placement_t[time_slot]))
                for r in range(len(params.NODE_LIST)):
                    for i in range(job.ps_placement_t[time_slot][r]):
                        ps_placement.append(params.NODE_LIST[r])
            worker_placement = []
            if time_slot in job.worker_placement_t.keys():
                self.logger.info("time_slot: " + str(time_slot))
                self.logger.info("job.worker_placement_t[time_slot]: " + str(job.worker_placement_t[time_slot]))
                for r in range(len(params.NODE_LIST)):
                    for i in range(job.worker_placement_t[time_slot][r]):
                        worker_placement.append(params.NODE_LIST[r])

            #self.logger.debug(self.name + " job.remain_workload: " + str(job.remain_workload))

            # send message to progressor to signal scheduling completion
            self.logger.debug(self.name + " sum(job.num_worker_t.values()) " + str(job.num_worker_t.values()))
            if sum(job.num_worker_t.values()) == job.total_worker:
                msg = (self.timer.get_clock(), 'scheduler', 'progressor', 'done', None)
                self.hub_queue.put(msg)
            
            if len(ps_placement) > 0 and len(worker_placement) > 0:  # and time_slot <= job.num_ps_t.keys()[-1]:

                # this may cause many ssh connections on a server and an error "ssh_exchange_identification: Connection closed by remote host"
                # to avoid this error, run 'echo "MaxStartups 100:10:200" | sudo tee -a /etc/ssh/sshd_config && sudo service ssh restart' on the server
                self.logger.debug(
                    self.name + ":: " + "job length of placement: " + str(len(ps_placement)) + ' ' + str(
                        len(worker_placement)))
                self.running_jobs.append(job)
                thread = threading.Thread(target=self.__run, args=(job, ps_placement, worker_placement, time_slot,))
                thread.start()
                thread_list.append(thread)
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

    def __run(self, job, ps_placement, worker_placement,time_slot):
        self.logger.debug(self.name + ":: " + job.name + ", num_ps: " + str(job.num_ps) + ", num_worker: " + str(
            job.num_worker) + ", ps placement: " + str(ps_placement) + ", worker placement: " + str(worker_placement))

        # set placement and start job
        # sys.exit()
        job.set_ps_placement(ps_placement)
        job.set_worker_placement(worker_placement)
        job.data_upload(time_slot)
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
        for job in self.rejected_jobs:
            job.delete(True)

    def _start_next_ts(self):
        # send message to timer to signal starting next timeslot
        msg = (self.timer.get_clock(), 'scheduler', 'timer', 'control', None)
        self.hub_queue.put(msg)
