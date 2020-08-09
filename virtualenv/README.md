## Virtual Environment

You need to install python3.7 or build it yourself before launching the virtual environment scripts. You might need `libffi-dev`.

Files in this folder:
* `do_make_virtualenv_setup3#.sh` -- run once to add virtualenv to the installed packages... or simply `source` this file.
* `do_make_virtualenv_use_3#.sh` -- run every time you want to use your environment. The env name is 'chatbot3#'

These files probably work best if run with `source`. Type `deactivate` to exit the virtualenv.

## Building Python3.7

This formula seems to work for Python3.6 also.

```
sudo apt-get install libssl-dev libbz2-dev libffi-dev libsqlite3-dev sqlite3 # other dev packages may be required
wget --no-check-certificate  https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz
tar xvzf Python-3.7.7.tgz 
cd Python-3.7.7
./configure --enable-optimizations --enable-loadable-sqlite-extensions
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

## Setup file:
This is the contents of the setup python 3.6 file. 
You may have to run these commands without the `sudo` or `--user` options.

```
sudo pip3 install --user virtualenv
sudo pip3 install --user virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
mkdir -p $WORKON_HOME
export VIRTUALENVWRAPPER_PYTHON=$(which python3.6)
source $(which virtualenvwrapper.sh)

mkvirtualenv chatbot36 --python $(which python3.6)
```