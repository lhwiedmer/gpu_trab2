



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
            for (int i = 0; i < h; i++) {
                if (input[ig] < limit[i]) {
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


__global__ void scan(unsigned int *hg, unsigned int *shg, unsigned int h){

}
