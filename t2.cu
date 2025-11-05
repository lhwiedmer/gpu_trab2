__global__ void blockAndGlobalHisto(unsigned int *hh, unsigned int *hg,
    unsigned int h, unsigned int *input, long long int nE, unsigned int min,
    unsigned int max) {
    extern __shared__ unsigned int histo[];
    extern __shared__ unsigned int limit[];
    //Incializa o vetor histo com 0
    if (threadIdx.x < h)
        histo[threadIdx.x] = 0;
    unsigned int tamFaixa = (max - min + h) / h; //Calcula o teto do numero de possiveis valores sobre o numero de faixas
    //Calcula o limite superior do intervalo de valores de cada faixa e guarda no vetor limit
    if (threadIdx.x < h)
        limit[threadIdx.x] = min + tamFaixa * (threadIdx.x + 1);
    unsigned int start = blockIdx.x * blockDim.x;
    unsigned int d = gridDim.x * blockDim.x;
    unsigned int ig;
    __syncthreads();
    //Preenche o histograma local
    while (start < nE) {
        ig = start + threadIdx.x;
        if (ig < nE) {
            unsigned int aux = input[ig];
            for (int i = 0; i < h; i++) {
                if (aux < limit[i]) {
                    atomicAdd(histo+i, 1);
                    break;
                }
            }
        }
        __syncthreads();
        start += d;
    }
    //Copia o histograma local do bloco para a linha corespondente de hh
    if (threadIdx.x < h)
        hh[threadIdx.x + blockIdx.x * h] = histo[threadIdx.x];
    //Calcula o histograma global hg
    if (threadIdx.x < h)
        atomicAdd(hg + threadIdx.x, histo[threadIdx.x]);
}

//Dispara 1 bloco com h threads e aloca um vetor de tamanho h na shared memory
__global__ void GlobalHistoScan(unsigned int *hg, unsigned int *shg, unsigned int h){
    extern __shared__ unsigned int XY[];
    if (threadIdx.x < h) {
        XY[threadIdx.x] = hg[threadIdx.x];
    }
    unsigned int numThreads = (h+1) >> 1;
    __syncthreads();
    for (unsigned int stride = 1;stride <= numThreads; stride = stride << 1) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < h)
            XY[index] += XY[index-stride];
        __syncthreads();
    }
    for (unsigned int stride = (numThreads+1) >> 1; stride > 0; stride = stride >> 1) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < h)
            XY[index + stride] += XY[index];
    }
    __syncthreads();
    if (threadIdx.x < h) {
        if (!threadIdx.x) {
            shg[threadIdx.x] = 0;
        } else {
            shg[threadIdx.x] = XY[threadIdx.x - 1];
        }
    }
}

//Dispara h blocos com lin threads e aloca um vetor de tamanho lin em shared memory
__global__ void verticalScanHH(unsigned int *hh, unsigned int *psv, unsigned int h, unsigned int lin){
    extern __shared__ unsigned int XY[];
    unsigned int ig = blockIdx.x + threadIdx.x * h;
    unsigned long long int nE = lin * h;
    if (ig < nE)
        XY[threadIdx.x] = hh[ig];
    __syncthreads();
    unsigned int numThreads = (lin+1) >> 1;
    for (unsigned int stride = 1;stride <= numThreads; stride = stride << 1) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < lin)
            XY[index] += XY[index-stride];
        __syncthreads();
    }
    for (unsigned int stride = (numThreads+1) >> 1; stride > 0; stride = stride >> 1) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < lin)
            XY[index + stride] += XY[index];
    }
    __syncthreads();
    if (ig < nE) {
        if (!threadIdx.x) {
            psv[ig] = 0;
        } else {
            psv[ig] = XY[threadIdx.x - 1];
        }
    }
}

__global__ void partitionKernel(unsigned int *hh, unsigned int *shg, unsigned int *psv,
                                unsigned int h, unsigned int *input, unsigned int *output,
                                unsigned long long int nE, unsigned int nMin, unsigned int nMax) {  
    extern __shared__ unsigned int hlsh[];
    extern __shared__ unsigned int limit[];
    __shared__ unsigned int parcial[1024];
    if (threadIdx.x < h) {
        hlsh[threadIdx.x] = shg[threadIdx.x];
        hlsh[threadIdx.x] += psv[threadIdx.x + blockIdx.x * h];
    }
    unsigned int tamFaixa = (nMax - nMin + h) / h; //Calcula o teto do numero de possiveis valores sobre o numero de faixas
    //Calcula o limite superior do intervalo de valores de cada faixa e guarda no vetor limit
    if (threadIdx.x < h)
        limit[threadIdx.x] = min + tamFaixa * (threadIdx.x + 1);
    unsigned int start = blockIdx.x * blockDim.x;
    unsigned int d = gridDim.x * blockDim.x;
    unsigned int ig;
    __syncthreads();
    while (start < nE) {
        ig = start + threadIdx.x;
        if (ig < nE) {
            unsigned int aux = input[ig];
            for (int i = 0; i < h; i++) {
                if (aux < limit[i]) {
                    parcial[atomicAdd(hlsh+i, 1)] = aux;
                    break;
                }
            }
        }
        __syncthreads();
        if (ig < nE) {
            output[ig] = parcial[threadIdx.x]; 
        }
        start += d;
    }
}
