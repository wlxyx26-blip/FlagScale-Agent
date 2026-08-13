###使用教程###

设备：Linux系统、GPU
说明:建议本地运行，将FlagScale-Agent源码和task任务放在不同文件夹，例如：

|--Test

  |--FlagScale-Agent
  
  |--task
  
    |--data
    
    |--Harbor_adapter
    
    |--imdb-bert-train
    
    |-- ....

1、安装harbor

apt install -y pipx

pipx install harbor --pip-args="-i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300"     #国内镜像

pipx ensurepath       #将路径添加到PATH 

source ~/.bashrc

harbor --version 

2、自定义接口

Harbor支持集成用户自定义的代理程序，当前已实现。 

3、测评任务
当前有两个测评任务，"imdb-bert-train" 和 "pytorch-model-recovery"
