#! /bin/bash

rm kungpao_camera_linux.o
rm kungpao_camera.so

gcc -fPIC -Wall -c -o kungpao_camera_linux.o kungpao_camera.c
gcc -shared -o kungpao_camera.so kungpao_camera_linux.o
