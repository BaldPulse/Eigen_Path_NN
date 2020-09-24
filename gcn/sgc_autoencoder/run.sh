#!/bin/bash

edge=4
path=2
nl=0.2
iter=500

rm -rf ./logs/*

while [ $edge -le 4 ]
do
    path=2
    while [ $path -le 5 ]
    do
	echo python3 sgc_autoencoder_keras.py --data sfg/100/shared_edges/nl_${nl}/${edge}/${path}_pth --tensorboard -n ${iter} --name ${edge}e_${path}p
	python3 sgc_autoencoder_keras.py --data sfg/100/shared_edges/nl_${nl}/${edge}/${path}_pth --tensorboard -n ${iter} --name ${edge}e_${path}p
	python3 sgc_autoencoder_keras.py --data sfg/100/shared_edges/nl_${nl}/${edge}/${path}_pth --tensorboard -n ${iter} --name ${edge}e_${path}p
	((path++))
    done
    rm -rf ../../results/shared_edges/nl_${nl}/${edge}e/*
    echo cp -r ./logs/* ../../results/shared_edges/nl_${nl}/${edge}e/
    cp -r ./logs/* ../../results/shared_edges/nl_${nl}/${edge}e/
    rm -rf ./logs/
    ((edge++))
done


