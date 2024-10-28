import time
import datetime
import os
import sys
import threading
import subprocess
import requests
import ast
import json 
import numpy as np 
import params
import random

class Job(object):
    def __init__(self, id, type, model_name, workload_id, dir_prefix, logger):
        # initialize a job
        # job type: eg., measurement-imagenet, i.e., category-dataset
        self.id = id
        self.type = type
        self.model_name = model_name
        self.workload_id = workload_id
        self.name = str(id) + '-' + type + '-' + model_name

        now = time.time()
        self.timestamp = str(datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d-%H:%M:%S'))
        self.dir = dir_prefix + self.name + '-' + self.timestamp + '/'

        self.logger = logger

        self.accept = 0
        self.num_ps = None
        self.num_ps_t = {}   # [time]  %
        self.ps_cpu = None
        self.ps_mem = None
        self.ps_bw = None

        self.num_worker = None
        self.num_worker_t = {}   #[time] %
        self.worker_cpu = None
        self.worker_mem = None
        self.worker_bw = None
        self.worker_gpu = None
        ##########################new add #########################
        self.P = None  
        self.params_size = None
        self.social_welfare = 0
        self.cost = 0
        
        self.ps_placement = None
        self.worker_placement = None
        
        self.ps_placement_t = {}   #[time][r]   ---->z_i^{r}(t)
        self.worker_placement_t = {}  #[time][r]   ----> g_i^{r}(t)
        
        #D_i^{r}
        self.train_data = []
        
        #[time][r]   --->D_t^{r}
        self.trained_data={}
        # tran_data[time][r][r']  --->h_t^{rr'}
        self.tran_data={}
        self.remain_workload = 0

        for x in range(1,params.T+1):#10 extra time slot for overflow
            self.trained_data[x]=[0 for i in range(len(params.NODE_LIST))]
            self.tran_data[x]=[[0 for i in range(len(params.NODE_LIST))] for i in range(len(params.NODE_LIST))]

        self.speed_list = []



        # [(epoch, batch)]
        self.progress_list = None
        self.ps_metrics = []
        self.worker_metrics = []
        self.ps_pods = []
        self.worker_pods = []

        self.kv_store_big_array_bound = str(1000*1000)
        self.ps_verbose = ''

        # for experiment
        self.arrival_slot = None
        self.arrival_time = None
        self.end_slot = None
        self.end_time = None
        self.status = 'initialized'
        self.progress = 0
        self.data_chunks = 0
        self.picture_size = 0

        # (num_ps, num_worker): speed
        self.training_speeds = dict()

        # epoch : validation_loss
        self.val_losses = dict()
        self.num_epochs = 0
        self.epoch_size = 0
        
    def set_ps_resources(self, num_ps, ps_cpu, ps_mem, ps_bw=''):
        # resource requirements of parameter servers
        self.num_ps = num_ps
        self.ps_cpu = ps_cpu
        self.ps_mem = ps_mem
        self.ps_bw = ps_bw

    def set_worker_resources(self, num_worker, worker_cpu, worker_mem, worker_bw='', worker_gpu='0'):
        # resource requirements of workers
        self.num_worker = num_worker
        self.worker_cpu = worker_cpu
        self.worker_mem = worker_mem
        self.worker_bw = worker_bw
        self.worker_gpu = worker_gpu

    def set_ps_placement(self, ps_placement):
        # the placement of parameter servers
        if isinstance(ps_placement, list):
            if len(ps_placement) == self.num_ps:
                self.ps_placement = ps_placement
            else:
                raise RuntimeError('ps_placement is not consistent with num_ps')
        else:
            raise TypeError('ps_placement is not a list')

    def set_worker_placement(self, worker_placement):
        # the placement of workers
        if isinstance(worker_placement, list):
            self.logger.info("length of worker-placement: " + str(len(worker_placement)) + "; num_worker: " + str(self.num_worker) )
            if len(worker_placement) == self.num_worker:
                self.worker_placement = worker_placement
            else:
                raise RuntimeError('worker_placement is not consistent with num_worker')
        else:
            raise TypeError('worker_placement is not a list')

    def _set_mount_dirs(self, type, host_workdir_prefix):
        # directories on hosts mounted to containers
        mount_dirs = []
        if type == 'ps':
            for i in xrange(self.num_ps):
                postfix = self.name + '-ps-' + str(i) + '/'
                mount_dir = host_workdir_prefix + postfix
                mount_dirs.append(mount_dir)
                cmd = 'ssh ' + self.ps_placement[i] + ' "sudo rm -rf ' + mount_dir + '; mkdir -p ' + mount_dir + '"'
                os.system(cmd)

        elif type == 'worker':
            for i in xrange(self.num_worker):
                postfix = self.name + '-worker-' + str(i) + '/'
                mount_dir = host_workdir_prefix + postfix
                mount_dirs.append(mount_dir)
                cmd = 'ssh ' + self.worker_placement[i] + ' "sudo rm -rf ' + mount_dir + '; mkdir -p ' + mount_dir + '"'
                os.system(cmd)
        return mount_dirs

    def set_container(self, image, script, work_dir, host_workdir_prefix, work_volume='k8s-mxnet-work-volume'):
        # container description
        self.image = image
        self.script = script
        self.work_dir = work_dir
        self.host_workdir_prefix = host_workdir_prefix
        self.work_volume = work_volume

    def set_data(self, hdfs_data, raw_data_dir, data_dir, host_data_dir, data_mounted=True, data_volume='k8s-mxnet-data-volume'):
        # data specification, if data not in local host, fetch from HDFS
        # dataset list including training data and validation data
        self.hdfs_data = hdfs_data
        self.raw_data_dir = raw_data_dir
        self.data_dir = data_dir
        self.host_data_dir = host_data_dir
        self.data_mounted = data_mounted
        self.data_volume = data_volume

    def set_train(self, prog, batch_size, kv_store, scale_bs=False, num_examples=0, params_size=0, picture_size=0, num_epochs=sys.maxint):
        self.prog = prog
        self.tot_batch_size = batch_size
        self.kv_store = kv_store
        self.scale_bs = scale_bs
        self.num_examples = num_examples
        self.params_size = params_size
        self.picture_size =picture_size
         
        # for unknown num_epochs, will update it in progressor with estimation
        self.num_epochs = num_epochs

    def __set_batch_size(self):
        # the batch size of each worker for sync training may be different
        self.logger.info("Job 154:: " + "starting to set epoch/batch size!")
        if 'async' in self.kv_store:
            self.logger.debug("Job 161:: " + "num-examples: " + str(self.num_examples))
            self.logger.debug("Job 161:: " + "tot-batch-size: " + str(self.tot_batch_size))
            self.epoch_size = self.num_examples / self.tot_batch_size
            self.batch_sizes = [str(self.tot_batch_size) for i in range(self.num_worker)]
            self.logger.debug("Job 161:: " + "epoch_size: " + str(self.epoch_size))
        elif 'sync' in self.kv_store:
            self.epoch_size = self.num_examples / self.tot_batch_size / self.num_worker
            self.batch_sizes = [str(self.tot_batch_size) for i in range(self.num_worker)]
            

        if self.kv_store == 'dist_async':
            self.batch_sizes = [str(self.tot_batch_size) for i in range(self.num_worker)]
        elif self.kv_store == 'dist_sync' or self.kv_store == 'dist_device_sync':
            # will change global batch size during training.
            if self.scale_bs:
                self.batch_sizes = [str(self.tot_batch_size) for i in range(self.num_worker)]
            else:
                avg_batch_size = self.tot_batch_size / self.num_worker
                rem_batch_size = self.tot_batch_size % self.num_worker
                batch_sizes = [avg_batch_size for i in range(self.num_worker)]
                for i in range(rem_batch_size):
                    batch_sizes[i] = batch_sizes[i] + 1
                self.batch_sizes = [str(i) for i in batch_sizes]

    def set_data_dist(self, length, data_chunks):
        self.data_chunks= data_chunks
        self.remain_workload = data_chunks
        L = [random.randint(10,100) for i in range(length) ] 
        for r in range(length):
            self.train_data.append(self.data_chunks * L[r] / sum(L))
        min_index = self.train_data.index(min(self.train_data))
        self.train_data[min_index] += (self.data_chunks - sum(self.train_data))
        

    def set_mxnet(self, kv_store_big_array_bound, ps_verbose=''):
        # set env MXNET_KVSTORE_BIGARRAY_BOUND
        self.kv_store_big_array_bound = str(kv_store_big_array_bound)
        self.ps_verbose = ps_verbose

    def __list_to_str(self, _listofstr):
        string = ''
        for i in xrange(len(_listofstr)):
            if i < len(_listofstr) - 1:
                string = string + _listofstr[i] + ','
            else:
                string = string + _listofstr[i]
        return string

    def _create(self):
        # create job definition, i.e., yaml file
        variables = {}
        variables['JOB_NAME'] = self.name

        variables['IMAGE'] = self.image
        variables['SCRIPT'] = self.script
        variables['PROG'] = self.prog
        variables['WORK_DIR'] = self.work_dir
        variables['PS_MOUNT_DIRS'] = self.__list_to_str(self.ps_mount_dirs)
        variables['WORKER_MOUNT_DIRS'] = self.__list_to_str(self.worker_mount_dirs)
        variables['WORK_VOLUME'] = self.work_volume
        variables['DATA_DIR'] = self.data_dir
        variables['DATA_MOUNT_DIR'] = self.host_data_dir
        variables['DATA_VOLUME'] = self.data_volume

        variables['NUM_PS'] = str(self.num_ps)
        variables['PS_CPU'] = str(self.ps_cpu)
        variables['PS_MEM'] = str(self.ps_mem) + "Gi"

        variables['NUM_WORKER'] = str(self.num_worker)
        variables['WORKER_CPU'] = str(self.worker_cpu)
        variables['WORKER_MEM'] = str(self.worker_mem) + "Gi"
        variables['WORKER_GPU'] = str(self.worker_gpu)

        variables['PS_PLACEMENT'] = self.__list_to_str(self.ps_placement)
        variables['WORKER_PLACEMENT'] = self.__list_to_str(self.worker_placement)

        variables['BATCH_SIZES'] = self.__list_to_str(self.batch_sizes)
        variables['KV_STORE'] = self.kv_store

        variables['MXNET_KVSTORE_BIGARRAY_BOUND'] = self.kv_store_big_array_bound
        variables['PS_VERBOSE'] = self.ps_verbose

        # copy template file
        self.jinja = self.dir + self.name + '.jinja'
        os.system("cp ../templates/k8s-mxnet-template.jinja " + self.jinja)

        # replace variables in jinja file
        temp_file = self.jinja + '.temp'
        for key, value in variables.items():
            os.system('sed -e "s@\$' + key + '@' + value + '@g" "' + self.jinja + '"' + ' > ' + temp_file)
            os.system('rm ' + self.jinja)
            os.system('mv ' + temp_file + ' ' + self.jinja)

        # generate yaml file
        self.yaml = self.dir + self.name + '.yaml'
        os.system("python ../templates/render-template.py " + self.jinja + " > " + self.yaml)

    def _read_data(self):
        # if not mounted from local host, then read data from HDFS
        if self.data_mounted:
            self.logger.info("Job 273:: " + "job read data from localhost, not from HDFS!")
            return
        if self.hdfs_data is None or self.hdfs_data == '':
            raise ValueError('data is not mounted from localhost and hdfs_data is not specified')
        
        # get training and validation data from HDFS
        thread_list = []
        temp_list = []
        for i in xrange(self.num_worker):
            node = self.worker_placement[i]
            if node not in temp_list:
                #self.logger.info("data_upload list:" + str(temp_list))
                for data in self.hdfs_data:
                    fn = data.split("/")[-1]
                    local_file = self.host_data_dir + fn
                    cmd = 'ssh ' + node + ' "/usr/local/hadoop-3.2.1/bin/hadoop fs -copyToLocal -f ' + data + ' ' + self.host_data_dir + '"'
                    thread_train = threading.Thread(target=(lambda cmd=cmd: os.system(cmd)), args=())
                    thread_train.start()
                    thread_list.append(thread_train)
                temp_list.append(node)
        self.logger.info("data_upload list:" + str(temp_list))
        for thread in thread_list:
            thread.join()           
  

    def _read_progress_stats(self):
        # get the job progress from each worker
        progress_fn = 'progress.txt'

        # create a new one each time, since the number of workers will change, hence the size of progress list
        self.progress_list = [(0,0) for i in xrange(self.num_worker)]
        self.val_loss_list = [(0,0) for i in xrange(self.num_worker)]
        thread_list = []
        for i in xrange(self.num_worker):
            node = self.worker_placement[i]
            local_file = self.worker_mount_dirs[i] + progress_fn
            cmd = "ssh " + node + " 'cat " + local_file + "'"
            #self.logger.info("Job 271:: " + "reading progress cmd: " + cmd)
            def run(self, cmd, i):
                try:
                    output = subprocess.check_output(cmd, shell=True)
                    counter = 0
                    self.logger.info("Job 276:: " + "reading progress " + output)
                    while output == '' or output == None:
                        output = subprocess.check_output(cmd, shell=True)
                        time.sleep(0.001 * (10 ** counter))
                        counter = counter + 1
                        if counter > 2:
                            break
                    if output is not None and output != '':
                        # should not be empty, even no progress, there should be initialization values written in files.
                        stat_dict = ast.literal_eval(output.replace('\n', ''))
                        if "progress" in stat_dict and "val-loss" in stat_dict:
                            self.progress_list[i] = stat_dict["progress"]

                            # it is a list of (epoch, loss)
                            self.val_loss_list[i] = stat_dict["val-loss"]
                        else:
                            self.logger.info("Job:: " + "progress output does not have progress or val-loss value")
                    else:
                        self.logger.info("Job:: " + "the progress output is empty.")
                except Exception as e:
                    self.logger.error("Job:: " + "_read_progress_stats: " + str(e) + " : " + output)

            thread = threading.Thread(target=run, args=(self, cmd, i))
            thread.start()
            thread_list.append(thread)
        for thread in thread_list:
            thread.join()

    def get_training_progress_stats(self):
        self._read_progress_stats()
        self.logger.info("Job 305:: " + "get training progress stats" )
        return (list(self.progress_list), list(self.val_loss_list))

    def _read_training_speed(self):
        # get the job training speed from each worker
        speed_fn = 'speed.txt'
        self.speed_list = [0 for i in xrange(self.num_worker)]
        thread_list = []
        for i in xrange(self.num_worker):
            node = self.worker_placement[i]
            local_file = self.worker_mount_dirs[i] + speed_fn
            '''
            cmd = 'scp ' + node + ':' + local_file + ' ' + self.dir # the new txt will replace the old one, no need to delete
            os.system(cmd)
            try:
                with open(self.dir+speed_fn, 'r') as fh:
                    stb_speed = float(fh.readline().replace('\n', '').split(' ')[1])
                    self.speed_list[i] = float('%.3f'%(stb_speed))
            except Exception as e:
                print e
                continue
            '''
            cmd = "ssh " + node + " 'cat " + local_file + "'"
            self.logger.debug("job 329:: " + "read training speed cmd: " + cmd)
            def run(self, cmd, i):
                try:
                    output = subprocess.check_output(cmd, shell=True)

                    # the other side is opening and writing the file, try again
                    counter = 0
                    while output == '' or output == None:
                        output = subprocess.check_output(cmd, shell=True)
                        time.sleep(0.001*(10**counter))
                        counter = counter + 1
                        if counter > 2:
                            self.logger.error("Job 341:: " + "_read_training_speed: read training speed timeout.")
                            return
                    stb_speed = float(output.replace('\n', '').split(' ')[1])
                    self.speed_list[i] = float('%.3f'%(stb_speed))
                    
                    ## training_speed gets from estimator     
                    self.P = self.speed_list[0]* params.TS_INTERVAL*100 /self.picture_size /self.num_epochs
                    self.logger.debug("job process capacity: " +str(self.P))
                except Exception as e:
                    self.logger.error("Job 346:: " + "_read_training_speed: " + str(e))

            thread = threading.Thread(target=run, args=(self, cmd, i))
            thread.start()
            thread_list.append(thread)
        for thread in thread_list:
            thread.join()

    def get_training_speed(self):
        self._read_training_speed()
        self.logger.info("Job 356:: " + "get training speed" )
        return list(self.speed_list)

    
    def __get_pods(self, task):
        """
        get the names of the pods belonging to the task

        NAME                                    READY     STATUS    RESTARTS   AGE
        1-measurement-imagenet-ps-0-mzv2z       1/1       Running   0          1m
        """
        if task == 'ps':
            self.ps_pods = []
        elif task == 'worker':
            self.worker_pods = []
        else:
            raise ValueError('task can only either be ps or worker!')
        cmd = 'kubectl get pods --selector=' + 'name=' + self.name + ',' + 'job=' + task + ' --namespace=default' + ' |grep ' + task #+ " |awk '{print $1}'"
        #cmd = 'kubectl get pods --selector=' + 'name=' + self.name + ',' + 'job=' + task + ' --namespace=default' + ' |grep ' + task
        output = subprocess.check_output(cmd, shell=True)
        self.logger.debug("Job 378:: " + "get pods cmd: " + cmd)
        #lines = output.split('\n')
        try:
            lines = output.split('\n')
            self.logger.debug("Job 383::" + lines)
        except:
            self.logger.debug("Job 385:: get lines error!")
        for line in lines:
            if len(line) > 0:
                words = line.split(' ')
                if task == 'ps':
                    self.ps_pods.append(words[0])
                else:
                   self.worker_pods.append(words[0])

    def _read_metrics(self):
        # get the metrics of the pods of this job
        # get ps/worker pods
        self.logger.debug(self.name + " starting to read pods' metrics!")
        self.__get_pods('ps')
        self.__get_pods('worker')
       
        # get heapster cluster ip
        # heapster               192.168.192.16    <none>        80/TCP              5d
        cmd = "kubectl get services --namespace=kube-system | grep metrics |awk '{print $3}'"
        heapster_cluster_ip = subprocess.check_output(cmd, shell=True).replace('\n','')
        if heapster_cluster_ip == 'ClusterIP':
            heapster_cluster_ip = 'xxxxxx'

        '''
        {
          "metrics": [
           {
            "timestamp": "2017-08-14T08:10:00Z",
            "value": 0
           }
          ],
          "latestTimestamp": "2017-08-14T08:10:00Z"
         }
        '''
        self.ps_metrics = []
        self.worker_metrics = []
        self.logger.debug(self.name + "::410 " + "heapster ip: " + heapster_cluster_ip)
        # cpu: milli core, mem: bytes, net: bytes/second
        metric_keys = ['cpu/usage_rate', 'memory/usage'] #'network/tx_rate', 'network/rx_rate'
        for pod in (self.ps_pods + self.worker_pods):
            pod_metrics = {}
            for metric_key in metric_keys:
                #url = 'http://10.201.148.165:6443/apis/metrics.k8s.io/v1beta1/namespaces/default/pods/' + pod
                url = 'https://10.201.148.165:6443/api/v1/namespaces/kube-system/services/https:metrics-server:/proxy/apis/metrics.k8s.io/v1beta1/namespaces/default/pods/' + pod
                #url = 'http://' + heapster_cluster_ip + '/apis/metrics.k8s.io/v1beta1/namespace/default/pods/' + pod + '/metrics/' + metric_key
                
                #self.logger.debug("Job 423:: " + "read metrics by request url: " + url)
                output = requests.get(url, verify=False).json()
                if metric_key == 'cpu/usage_rate':
                    str_cpu = output['containers'][-1]['usage']['cpu']
                    metric_value = int(str_cpu.replace('n','')) #exclude the cpu cores's unit XXXXn                    
                    #self.logger.debug("Job 438:: " + "cpu/usage_rate " + str(metric_value))
                else:
                    str_mem = output['containers'][-1]['usage']['memory'] # -1 represent the location of containers's dict
                    metric_value = int(str_mem.replace('Ki','')) #exclude the memory's unit XXXXKi
                    #self.logger.debug("Job 446:: " + "memory " + str(metric_value))

                pod_metrics[metric_key] = metric_value
            if pod in self.ps_pods:
                self.ps_metrics.append(pod_metrics)
                #self.logger.debug("Job 452:: " + "ps_metrics " + str(ps_metrics))
            else:
                self.worker_metrics.append(pod_metrics)
                #self.logger.debug("Job 455:: " + "woker_metrics " + str(woker_metrics))
        self.logger.debug("Job 456:: " + "finish read metrics! ")
    
    def get_metrics(self):
        self._read_metrics()
        self.logger.info("Job 515:: " + "get metrics")
        return (list(self.ps_metrics), list(self.worker_metrics))

    def get_workload_t(self,t):
        amount_data_t = 0
        amount_data_t += int(sum(self.trained_data[t])) + int(sum(map(sum, self.tran_data[t])))
        self.remain_workload -= amount_data_t

    def data_upload(self,time_slot):
        #total amount training data = trained_data[t] + tran_data[t]
        pre_trained_data_size = 0
        for t in range(self.arrival_slot,time_slot):
            pre_trained_data_size += int(sum(self.trained_data[t]))
            for r in range(len(params.NODE_LIST)):
                pre_trained_data_size += int(sum(self.tran_data[t][r]))

        trained_data_size_t = int(sum(self.trained_data[time_slot]))
        for r in range(len(params.NODE_LIST)):
            trained_data_size_t += int(sum(self.tran_data[time_slot][r]))

        self.logger.info(self.name + " 520:: starting to updaload training data of Time Slot " + str(time_slot))
        job_type = str(self.type[self.type.find("-",) + 1 :])
        '''
        make unit of data_chunks 
        '''
        #cmd = 'python3 ' + self.raw_data_dir + job_type + '/'  + 'im2rec.py --recursive --list --chunks=' + num_chunks_t +  ' ' + job_type + ' ' + self.raw_data_dir + job_type +'/class/'
        #os.system(cmd)
        #job_type --> imagenet job.model_name ---> resnet-50 
        # tmp_str ---> ~/training_data/imagenet/vgg-16/
        tmp_str = self.raw_data_dir + self.model_name + '/' + job_type
        cmd= 'touch ' + tmp_str + '_train.lst'
        # [\sum(pre num_chunks-1),\sum(pre num_chunks)+num_chunks_per_ts -1 ] 
        for i in range(pre_trained_data_size, trained_data_size_t):
            cmd= 'cat ' + self.raw_data_dir + job_type + '/' + job_type + '_' + str(i) + '.lst >> '+ tmp_str + '_train.lst'
            os.system(cmd)
      
        cmd= 'python3 ' + self.raw_data_dir + self.model_name + '/' + 'im2rec.py --recursive  --chunks=1 ' + tmp_str + '_train ' + self.raw_data_dir + job_type + '/class/'
        os.system(cmd)
        for data in self.hdfs_data:
            cmd = '/usr/local/hadoop-3.2.1/bin/hadoop fs -rm ' + data 
            os.system(cmd)
        cmd = '/usr/local/hadoop-3.2.1/bin/hadoop fs -put  ' + tmp_str + '_train.rec' + ' ' + '/k8s-mxnet/' + job_type + '/' + self.model_name + '/'
        os.system(cmd)
        cmd= 'rm -rf ' + tmp_str + '_train.*'
        os.system(cmd)
        

    def start(self):
        # start the job in k8s
        self.logger.info("Job 520:: " + "starting job " + self.name + "...")

        # job working dir
        os.system('mkdir -p ' + self.dir)
        self.ps_mount_dirs = self._set_mount_dirs('ps', self.host_workdir_prefix)  # ps container mount
        self.worker_mount_dirs = self._set_mount_dirs('worker', self.host_workdir_prefix)  # worker container mount
        self.__set_batch_size()

        # create job yamls
        self._create()

        # prepare data
        self._read_data()

        # start pods in k8s
        subprocess.check_output("kubectl create -f " + self.yaml, shell=True)

    def delete(self, del_all=False):
        """delete the job.
        Parameters
        ----------
        del_all: whether to delete all, including histories.
        """

        # shutdown job in k8s
        try:
            fh = open(self.yaml, 'r')
        except Exception as e:
            self.logger.error(" Failed to open " + self.yaml + ": " + str(e))
            return

        yamls = fh.read().split('---\n')
        fh.close()

        temp_dir = self.dir + 'temp/'
        os.system('mkdir -p ' + temp_dir)

        thread_list = []
        for i in range(len(yamls)):
            if len(yamls[i]) <= 1:
                # skip invalid
                continue
            name = temp_dir + str(i) + '.yaml'
            with open(name, 'w') as fh:
                fh.write(yamls[i])
            thread = threading.Thread(target=(lambda name=name: subprocess.check_output('kubectl delete -f ' + name, shell=True)), args=())
            thread.start()
            thread_list.append(thread)

        for thread in thread_list:
            thread.join()
        os.system('rm -rf ' + temp_dir)

        # in case not delete all
        subprocess.check_output('kubectl delete jobs --selector=name=' + self.name, shell=True)

        if not del_all:
            return

        # remove mounted dirs on hosts
        thread_list = []
        for i in xrange(self.num_worker):
            node = self.worker_placement[i]
            worker_mount_dir = self.worker_mount_dirs[i]
            cmd = 'timeout 10 ssh ' + node + ' "sudo rm -r ' + worker_mount_dir + '"'
            thread = threading.Thread(target=(lambda cmd=cmd: os.system(cmd)), args=())
            thread.start()
            thread_list.append(thread)

        for i in xrange(self.num_ps):
            node = self.ps_placement[i]
            ps_mount_dir = self.ps_mount_dirs[i]
            cmd = 'timeout 10 ssh ' + node + ' "sudo rm -r ' + ps_mount_dir + '"'
            thread = threading.Thread(target=(lambda cmd=cmd: os.system(cmd)), args=())
            thread.start()
            thread_list.append(thread)
        for thread in thread_list:
            thread.join()

        # delete job working dir
        subprocess.check_output("rm -rf " + self.dir, shell=True)


