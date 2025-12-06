#!/bin/bash

set -e  # stop on errors

FLAGS="-std=c++17 -O3"
SOURCES="l2lsh_cartesian.cpp utils.cpp"

INCLUDES=""
INCLUDES+=" -I$HOME/cpp_lib/or-tools/install_make/include"
INCLUDES+=" -I/usr/local/include"
INCLUDES+=" -I/usr/include/eigen3"

LIBS=""
LIBS+=" -L/usr/local/lib -lstag"
LIBS+=" -L$HOME/cpp_lib/or-tools/install_make/lib -lortools -lpthread"

OUTPUT="l2lsh_cartesian_cpp"

# g++ -E -H l2lsh_cartesian.cpp $FLAGS $INCLUDES > /dev/null

echo "Compiling..."
g++ $SOURCES $FLAGS $INCLUDES $LIBS -o $OUTPUT

echo "Build successful: ./$OUTPUT"

echo "Executing..."
DATASET=$1
c=$2
r=$3
w=$4
DELTA=${5-0.1}
./$OUTPUT "$DATASET" "$c" "$r" "$w" "$DELTA"
rm "$OUTPUT"
