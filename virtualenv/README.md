## Virtual Environment

You need to install python3.7 or build it yourself before launching the virtual environment scripts.

Files in this folder:
* `do_make_virtualenv_bash.sh` -- run once to add a few lines to your bash file that will let you run a virtualenv from the terminal easily.
* `do_make_virtualenv_setup.sh` -- run once to add virtualenv to the installed packages
* `do_make_virtualenv_use.sh` -- run every time you want to use your environment. The env name is 'chatbot'

## Building Python3.7

```
sudo apt-get install libssl-dev # other dev packages may be required
wget --no-check-certificate  https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz
tar xvzf Python-3.7.7.tgz 
cd Python-3.7.7
./configure --enable-optimizations
sudo make 
sudo make altinstall
```

## ubuntu update-alternatives
This might not be useful if you use virtual environment above.
```
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.7 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
# run below and select python3.7 from the list
sudo update-alternatives --config python3
```
