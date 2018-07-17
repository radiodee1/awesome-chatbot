#!/usr/bin/env bash

cd raw

if [ -f GoogleNews-vectors-negative300.bin.gz ] ; then
    echo "Found w2v file"
    echo 'Unzip will take some time!'
    gunzip GoogleNews-vectors-negative300.bin.gz

else
    if [ -f GoogleNews-vectors-negative300.bin ] ; then
        echo 'Uncompressed file found!'
        exit
    fi
    echo "The w2v file was not found!!"
    echo "This script will open the google-drive page that lets you donwload the w2v file."
    echo "Copy this file to the raw directory and run this script again!"
    echo "The download is 1.6 Gig. The uncompressed file size is: 3.4 Gig."

    xdg-open https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
fi

