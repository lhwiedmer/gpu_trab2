host=`hostname`
echo $host

if [ $host = "nv00" ]; then
   echo "compilando para maquina nv00."
   nvcc  -DTHREADS_PER_BLOCK=1024U  --std=c++14 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ -o mppSort t2.cu 
fi

if [ $host = "orval" ]; then
   echo "compilando para maquina orval."
   xx
   nvcc  -DTHREADS_PER_BLOCK=1024U  -arch sm_50 --allow-unsupported-compiler  -std=c++17 -Xcompiler=-std=c++17 -ccbin /usr/bin/g++-12 -o mppSort t2.cu 
fi