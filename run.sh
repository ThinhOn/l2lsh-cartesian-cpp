##!/bin/bash

set -e  # stop on errors

FLAGS="-std=c++17 -O3"
SOURCES="l2lsh_cartesian.cpp utils.cpp"

INCLUDES=""
INCLUDES+=" -I/usr/local/include"
INCLUDES+=" -I/usr/include/eigen3"

LIBS="-L/usr/local/lib -lstag"
OUTPUT="l2lsh_cartesian_cpp"

echo "Compiling..."
g++ $SOURCES $FLAGS $INCLUDES $LIBS -o $OUTPUT

echo "Build successful: ./$OUTPUT"

echo "Executing..."
DATASET=$1
c=$2
r=$3
w=$4
./$OUTPUT $DATASET $c $r $w
rm $OUTPUT