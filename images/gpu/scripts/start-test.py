import sys
import os
import time
import requests
import socket
import subprocess
import threading
import logging


logging.basicConfig(level=logging.INFO,	format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

ROLE = os.getenv("ROLE") #worker, ps or scheduler

# exported by k8s
HOST_NAME = os.getenv("HOSTNAME")
HOST_IP = socket.gethostbyname(HOST_NAME)
NUM_WORKER = os.getenv("DMLC_NUM_WORKER")
NUM_SERVER = os.getenv("DMLC_NUM_SERVER")

USE_DIST_KVSTORE = os.getenv("USE_DIST_KVSTORE")

APISERVER = "https://10.201.148.165:6443"
API = "/api/v1/namespaces/"
NAMESPACE = "default"
JOB_SELECTOR = "labelSelector=name="

# export via env
JOB_NAME = os.getenv("JOB_NAME") # JOB_NAME get from job.py's self.name

# the python main file starting training
PROG = os.getenv("PROG")
WORK_DIR = os.getenv("WORK_DIR")
BATCH_SIZE = os.getenv("BATCH_SIZE")
KV_STORE = os.getenv("KV_STORE")


'''
Get all pods of this job
'''


def get_podlist():
    pod = API + NAMESPACE + "/pods?"
    url = APISERVER + pod + JOB_SELECTOR + JOB_NAME 
    #url =https://10.201.148.165:6443/api/v1/namespaces/default/pods?labelSelector=name=1-experiment-mnist-resnext-110
    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    '''
    if os.path.isfile(token_path):
        token = open(token_path, 'r').read()
        bearer = "Bearer " + token
        headers = {"Authorization": bearer}
        return requests.get(url, headers=headers, verify=False).json()
    else:
        return requests.get(url,verify=False).json()
    '''
    return requests.get(url,verify=False).json()

'''
check whether all pods are running
'''


def is_all_running(podlist):
    require = len(podlist["items"])
    running = 0
    for pod in podlist["items"]:
        if pod["status"]["phase"] == "Running":
            running += 1
    logging.info("waiting for pods running, require:" + str(require) + ", running:" + str(running))
    if require == running:
        return True
    else:
        return False


'''
get pod <ip, id> mapping
'''


def get_map(podlist):
    global SCHEDULER_IP
     
    IPs = []
    for pod in podlist["items"]:
        IPs.append(pod["status"]["podIP"])
    logging.info("IPs before sorting")
    print(IPs)
    IPs.sort()
    logging.info("IPs after sorting")
    print(IPs)
    SCHEDULER_IP = str(IPs[0])
    map = {}
    for i in range(len(IPs)):
        map[IPs[i]] = i
    return map


def start_scheduler(cmd, env):
    logging.info("starting scheduler ...")

    # not in conflict with 'server' since they start in different time
    env['DMLC_ROLE'] = 'scheduler'
    scheduler = threading.Thread(target=(lambda: subprocess.check_call(cmd, env=env, shell=True)), args=())
    scheduler.setDaemon(True)
    scheduler.start()


def main():
    global ROLE

    logging.info("starting script ...")

    # interprete command
    cmd = "cd " + WORK_DIR + "../ && " + PROG
    if BATCH_SIZE is not None and BATCH_SIZE != '':
        cmd = cmd + " " + "--batch-size" + " " + BATCH_SIZE
    if KV_STORE is not None and KV_STORE != '':
        cmd = cmd + " " + "--kv-store" + " " + KV_STORE
    logging.info("cmd: " + cmd)
    
    env = os.environ.copy()
    if 'dist' in KV_STORE:  ###########################
        logging.info("Distributed training: " + KV_STORE)

        # check pod status
        logging.info("starting to get pod ...")
        podlist = get_podlist()
        logging.debug(str(podlist))


        while not is_all_running(podlist):
            time.sleep(1)
            podlist = get_podlist()
        logging.info("already confirm all pods in podlist are running!")

        
        map = get_map(podlist)
        logging.info(str(map))

        # the scheduler runs on the first node
        SCHEDULER_PORT = "6060"
        logging.info("scheduler IP: " + SCHEDULER_IP + ", scheduler port: " + SCHEDULER_PORT)
        env['DMLC_PS_ROOT_URI'] = SCHEDULER_IP
        env['DMLC_PS_ROOT_PORT'] = SCHEDULER_PORT
        env['DMLC_NUM_WORKER'] = NUM_WORKER
        env['DMLC_NUM_SERVER'] = NUM_SERVER
        # env['PS_VERBOSE'] = '2'

        logging.info("self role: " + ROLE + " self IP: " + HOST_IP)
        if SCHEDULER_IP  == HOST_IP:
            logging.info("master: start initialization ...")
            start_scheduler(cmd, env.copy())

        # start ps/worker
        if ROLE == "ps":
            ROLE = "server"
        env['DMLC_ROLE'] = ROLE
    #logging.debug(str(env['USE_DIST_KVSTORE']))
    #logging.debug(str(env['DMLC_ROLE']))
    subprocess.check_call(cmd, env=env, shell=True) #########
    logging.info("Task finished successfully!")


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print "Description: MXNet start script in k8s cluster"
        print "Usage: python start.py"
        sys.exit(1)
    main()

