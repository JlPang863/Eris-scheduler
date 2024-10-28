# build and push
docker build -t xxx/k8s-mxnet-gpu:latest -f k8s-mxnet-gpu.Dockerfile .
docker push xxx/k8s-mxnet-gpu:latest #sign in and push new images to your docker ID's repo

'''
docker build -t truemanlife/k8s-mxnet:release -f k8s-mxnet-gpu.Dockerfile .    
docker push truemanlife/k8s-mxnet:release
'''

####
#progress-monitor.py has modified
'''
docker build -t truemanlife/k8s-experiment:release -f k8s-mxnet-gpu.Dockerfile .    
docker push truemanlife/k8s-experiment:release  
'''