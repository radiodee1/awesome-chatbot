#!/bin/bash

echo $#
echo $@


zip -r tens_2_tens.zip saved/t2t_trained_model/export/* data/t2t_data/vocab* $@

echo "use this script with any savable files as the parameter."