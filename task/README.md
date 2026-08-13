#安装harbor

apt install -y pipx
pipx install harbor --pip-args="-i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300"  #国内镜像
pipx ensurepath  #将路径添加到PATH 
source ~/.bashrc
harbor --version

