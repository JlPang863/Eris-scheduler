wget https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
tar -xvzf Python-2.7.18.tgz
cd Python-2.7.18/
./configure --prefix=$HOME/.local
make
make install
python2 -V

pip2 install numpy
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
~/.local/bin/python2 get-pip.py --user
export PATH=$HOME/.local/bin:$PATH
source ~/.bashrc
pip2 -V

pip2 install numpy
pip2 install scipy --user
pip2 install PyYAML==5.3.1 --user
pip2 install requests --user