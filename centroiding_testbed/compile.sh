#!/bin/bash

# clean:
rm ./centroiding.o
rm ./centroiding.so

# compile:
g++ -Wall -fPIC -c -o ./centroiding.o centroiding.cpp -fopenmp 
g++ -shared -o ./centroiding.so ./centroiding.o
