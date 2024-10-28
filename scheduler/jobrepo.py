import params


#job_repos = [('experiment-mnist','resnet-110')]
#job_repos = [('experiment-caltech101','lenet')]
#job_repos = [('experiment-cifar10', 'resnet-101')]
#job_repos = [('experiment-tiny-imagenet','alexnet')]

job_repos = [('experiment-cifar10', 'resnet-50'), ('experiment-cifar10', 'resnet-101'),
             ('experiment-caltech101','lenet'),  ('experiment-caltech101','googlenet'),
             ('experiment-tiny-imagenet','inception-bn'), ('experiment-tiny-imagenet','alexnet')]
'''
'''
'''
network
#inception_bn inception_v3 googlenet vgg alexnet resnet50 mlp lenet
'''


def set_config(job):
    is_member = False
    for item in job_repos:
        if job.type == item[0] and job.model_name == item[1]:
            is_member = True

    if not is_member:
        raise RuntimeError

    if 'cifar10' in job.type:
        if 'resnet-50' in job.model_name:
            _set_resnet50_cifar10_job(job)
        else:
            _set_resnet101_cifar10_job(job)
    elif 'caltech101' in job.type:
        if 'googlenet' in job.model_name:
            _set_googlenet_caltech101_job(job)
        else:
            _set_lenet_caltech101_job(job)

    elif 'tiny-imagenet' in job.type:
        if 'inception-bn' in job.model_name:
            _set_inception_tiny_imagenet_job(job)
        else:
            _set_alexnet_tiny_imagenet_job(job)
    else:
        _set_resnet110_mnist_job(job)
        #raise RuntimeError


'''
ResNet-50_Cifar10
'''
def _set_resnet50_cifar10_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 1
    ps_mem = 4
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 4
    worker_gpu = 1   #allocated gpu
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_cifar10.py --network resnet --num-layers 50 --model-prefix /data/ --disp-batches 25 \
    --num-epochs 20 --data-train /data/cifar10_train.rec'
    kv_store = 'dist_sync'#distributed training
    #kv_store = 'async'#local training
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    #parameters of data_chunks
    picture_size = 1850
    params_size = 97.4
    data_chunks = 27

    job.set_data_dist(len(params.NODE_LIST), data_chunks=data_chunks) # set data distribution among sites
    job.set_train(prog=prog, batch_size=32, kv_store=kv_store, num_examples=50000, num_epochs=60, params_size=params_size, picture_size=picture_size,scale_bs=True)
    hdfs_data = ['/k8s-mxnet/cifar10/resnet-50/cifar10_train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/cifar10/resnet-50/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data, raw_data_dir=raw_data_dir, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')



'''
ResNet-101_Cifar10
'''
def _set_resnet101_cifar10_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 1
    ps_mem = 2
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 2
    worker_gpu = 1   #allocated gpu
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_cifar10.py --network resnet --num-layers 110  --model-prefix /data/ --disp-batches 25 \
    --num-epochs 20 --data-train /data/cifar10_train.rec'
    kv_store = 'dist_sync'
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    #parameters of data_chunks
    picture_size = 1850
    params_size = 169.9
    data_chunks = 27
    job.set_data_dist(len(params.NODE_LIST), data_chunks=data_chunks) # set data distribution among sites
    job.set_train(prog=prog, batch_size=32, kv_store=kv_store, num_examples=50000, num_epochs=20, params_size=params_size, picture_size=picture_size,scale_bs=True)
    hdfs_data = ['/k8s-mxnet/cifar10/resnet-101/cifar10_train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/cifar10/resnet-101/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data, raw_data_dir=raw_data_dir, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')




'''
Lenet_Caltech101
'''
def _set_lenet_caltech101_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 1
    ps_mem = 5
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 5
    worker_gpu = 2
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)
    #--network vgg --num-layers 16
    prog = 'python train_imagenet.py --network lenet --model-prefix /data/ --disp-batches 20 --num-epochs 20 --data-train /data/caltech101_train.rec'
    kv_store = 'dist_sync'

    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    #parameters of data_chunks
    picture_size = 80
    params_size = 528
    data_chunks = 115
    job.set_data_dist(len(params.NODE_LIST), data_chunks=data_chunks) # set data distribution among sites
    job.set_train(prog=prog, batch_size=8, kv_store=kv_store,num_examples=9145, num_epochs=20,params_size=params_size,picture_size=picture_size, scale_bs=True)
    hdfs_data = ['/k8s-mxnet/caltech101/lenet/caltech101_train.rec']#, '/k8s-mxnet/caltech101/Caltech101_val.rec'
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/caltech101/lenet/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data, raw_data_dir=raw_data_dir, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
Googlenet_Caltech101
'''
def _set_googlenet_caltech101_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 1
    ps_mem = 4
    ps_bw = 0
    worker_cpu = 3
    worker_mem = 4
    worker_gpu = 2
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_imagenet.py --network googlenet --disp-batches 20 --num-epochs 20 --data-train /data/caltech101_train.rec '
    #kv_store = 'dist_sync'
    kv_store = 'async'
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    #parameters of data_chunks
    picture_size = 80
    params_size = 25.9
    data_chunks = 115
    job.set_data_dist(len(params.NODE_LIST), data_chunks=data_chunks) # set data distribution among sites

    job.set_train(prog=prog, batch_size=8, kv_store=kv_store, num_examples=9145, num_epochs=20,params_size=params_size,picture_size=picture_size, scale_bs=True)
    hdfs_data = ['/k8s-mxnet/caltech101/googlenet/caltech101_train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/caltech101/googlenet/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data, raw_data_dir=raw_data_dir, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')





'''
Inception-BN_ImageNet
'''
def _set_inception_imagenet_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 1
    ps_mem = 4
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 4
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'     ######################
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_imagenet.py --network inception-bn --model-prefix /data/ --disp-batches 5 --num-epochs 20 --data-train /data/imagenet_train.rec'
    kv_store = 'dist_sync'
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    #parameters of data_chunks
    picture_size = 120
    params_size = 42.9
    data_chunks = 60
    job.set_data_dist(length=len(params.NODE_LIST), data_chunks=data_chunks) # set data distribution among sites

    job.set_train(prog=prog, batch_size=8, kv_store=kv_store, num_examples=7200, num_epochs=20,params_size=params_size, picture_size=picture_size,scale_bs=True)
    hdfs_data = ['/k8s-mxnet/imagenet/inception-bn/imagenet_train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/imagenet/inception-bn/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data, raw_data_dir=raw_data_dir, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')



'''
alexnet_ImageNet
'''
def _set_alexnet_imagenet_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 2
    ps_mem = 4
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 4
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)
    #--network alexnet
    prog = 'python train_imagenet.py --network alexnet  --model-prefix /data/ --disp-batches 2 --num-epochs 20 --data-train /data/imagenet_train.rec'
    kv_store = 'dist_sync'
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    #parameters of data_chunks
    picture_size = 120
    params_size = 237.9
    data_chunks = 60
    job.set_data_dist(length=len(params.NODE_LIST), data_chunks=data_chunks) # set data distribution among sites

    job.set_train(prog=prog, batch_size=8, kv_store=kv_store, num_examples=7200, num_epochs=20, params_size=params_size,picture_size=picture_size, scale_bs=True)
    hdfs_data = ['/k8s-mxnet/imagenet/alexnet/imagenet_train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/imagenet/alexnet/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data, raw_data_dir=raw_data_dir, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
Inception-BN_tiny-ImageNet
'''
def _set_inception_tiny_imagenet_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 3
    ps_mem = 3
    ps_bw = 0
    worker_cpu = 3
    worker_mem = 6
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'     ######################
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_imagenet.py --network inception-bn --model-prefix /data/ --disp-batches 20 --num-epochs 20 --data-train /data/tiny-imagenet_train.rec'
    #kv_store = 'dist_sync'
    kv_store = 'async'
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    #parameters of data_chunks
    picture_size = 120
    params_size = 42.9
    data_chunks = 60
    job.set_data_dist(length=len(params.NODE_LIST), data_chunks=data_chunks) # set data distribution among sites

    job.set_train(prog=prog, batch_size=8, kv_store=kv_store, num_examples=7200, num_epochs=20,params_size=params_size, picture_size=picture_size,scale_bs=True)
    hdfs_data = ['/k8s-mxnet/tiny-imagenet/inception-bn/tiny-imagenet_train.rec'] #tiny-imagenet
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/tiny-imagenet/inception-bn/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data, raw_data_dir=raw_data_dir, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')



'''
alexnet_tiny-ImageNet
'''
def _set_alexnet_tiny_imagenet_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 2
    ps_mem = 4
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 4
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)
    #--network alexnet
    prog = 'python train_imagenet.py --network alexnet  --model-prefix /data/ --disp-batches 2 --num-epochs 20 --data-train /data/tiny-imagenet_train.rec'
    kv_store = 'dist_sync'
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    #parameters of data_chunks
    picture_size = 120
    params_size = 237.9
    data_chunks = 60
    job.set_data_dist(length=len(params.NODE_LIST), data_chunks=data_chunks) # set data distribution among sites

    job.set_train(prog=prog, batch_size=8, kv_store=kv_store, num_examples=7200, num_epochs=20,params_size=params_size, picture_size=picture_size,scale_bs=True)
    hdfs_data = ['/k8s-mxnet/tiny-imagenet/alexnet/tiny-imagenet_train.rec'] #tiny-imagenet
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/tiny-imagenet/alexnet/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data, raw_data_dir=raw_data_dir, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')




#########################################################################################################################
# Add more examples for testing

'''
ResNet-110_Mnist
'''
def _set_resnet110_mnist_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 2
    ps_mem = 5
    ps_bw = 1
    worker_cpu = 2
    worker_mem = 5
    worker_gpu = 1     #allocated gpu
    worker_bw = 1

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'truemanlife/k8s-experiment:v34'      ######################
    #image = 'bellamn/k8s-mxnet-gpu:latest'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training.log, progress.txt, speed.txt and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/mxnet/example/image-classification/data/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_mnist.py '
    #kv_store = 'dist_sync'
    kv_store = 'async'
    #prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=128, kv_store=kv_store, num_examples=60000, num_epochs=20, scale_bs=True)
    data_dir = '/data/'
    #hdfs_data = ['/k8s-mxnet/mnist/mnist_train.rec']
    hdfs_data = ['/k8s-mxnet/mnist/resnet-101/fashion_train.rec'] #no need to use hdfs
    host_data_dir = '/data/mxnet-data/mnist/'
    raw_data_dir = '~/training_data/' #the raw data that need to create .rec file
    job.set_data(hdfs_data=hdfs_data,raw_data_dir=raw_data_dir,  data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=False) # data_mounted determine whether to read deta from HDFS or not(True:localhost)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')
