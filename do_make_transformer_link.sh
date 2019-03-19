echo "use this to make link to movie or reddit corpus"
echo
echo "enter path to corpus as first parameter."

SIMPLE_PATH=$PWD/data/t2t_data/
DATA=raw.txt
START=$PWD

echo
echo $SIMPLE_PATH
echo $START
echo $1


ln -s $START/$1 $SIMPLE_PATH/$DATA

ls -hal $SIMPLE_PATH