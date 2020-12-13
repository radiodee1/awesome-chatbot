## Alternative for Google Speech To Text: Pocketsphinx

For this prerequisite we built pocketsphinx from source.

From the apt repository 'swig' was required.
````
git clone --recursive https://github.com/bambocher/pocketsphinx-python
cd pocketsphinx-python
python setup.py install
````

To use the code in this directory you need to set an environment variable before calling any of the `do_launch_` command scripts from the root of this repository.

This environment variable is passed to the `game.py` script found in the `bot` folder.
It is called `SPEECH_RECOGNITION` and can have the value `google` or `sphinx`.
If the value passed is `sphinx` the `game.py` script calls the code in this folder for speech recognition. 

At this writing the code has not been tested on the Raspberry Pi.

sphinxbase is at: 
````
wget https://sourceforge.net/projects/cmusphinx/files/sphinxbase/5prealpha/sphinxbase-5prealpha.tar.gz/download -O sphinxbase.tar.gz
https://github.com/bambocher/pocketsphinx-python/issues/23
````