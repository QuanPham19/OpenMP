cmake -S . -B build
cmake --build build
./build/vector_add

# lscpu | grep -i cache
# free -h