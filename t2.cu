/**
 * @brief Implementação de mppSort em CUDA.
 * @author Luiz Henrique Murback Wiedmer
 * @author Eduardo Giehl
 */

// C
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <limits>
#include <vector>

// CUDA e thrust
// CUDA e thrust
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#define SHARED_SIZE_LIMIT 1024

// --- Macro de verificação de erro CUDA ---
static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define CUDA_CHECK( err ) (HandleError( err, __FILE__, __LINE__ ))



__global__ void blockAndGlobalHisto(unsigned int *hh, unsigned int *hg,
    unsigned int h, unsigned int *input, long long int nE, unsigned int min,
    unsigned int max) {
    extern __shared__ unsigned int shared_mem[];

    unsigned int *histo = shared_mem;       // hlsh usa os primeiros 'h' elementos
    unsigned int *limit = &shared_mem[h];
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
    __syncthreads();
    //Copia o histograma local do bloco para a linha corespondente de hh
    if (threadIdx.x < h)
        hh[threadIdx.x + blockIdx.x * h] = histo[threadIdx.x];
    //Calcula o histograma global hg
    if (threadIdx.x < h)
        atomicAdd(hg + threadIdx.x, histo[threadIdx.x]);
}

//Dispara 1 bloco com h threads e aloca um vetor de tamanho h na shared memory
__global__ void GlobalHistoScan(unsigned int *hg, unsigned int *shg, unsigned int h,
                                unsigned int pot){
    extern __shared__ unsigned int XY[];
    if (threadIdx.x < h) {
        XY[threadIdx.x] = hg[threadIdx.x];
    } else if (threadIdx.x < pot) {
        XY[threadIdx.x] = 0;
    }
    __syncthreads();
    for (unsigned int stride = 1;stride <= pot/2; stride = stride * 2) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < pot)
            XY[index] += XY[index-stride];
        __syncthreads();
    }
    for (unsigned int stride = pot/2; stride > 0; stride = stride / 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < pot)
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
    for (unsigned int stride = 1;stride <= numThreads; stride = stride * 2) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < lin)
            XY[index] += XY[index-stride];
        __syncthreads();
    }
    for (unsigned int stride = (numThreads+1) >> 1; stride > 0; stride = stride / 2) {
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
    extern __shared__ unsigned int shared_mem[];

    unsigned int *hlsh = shared_mem;       // hlsh usa os primeiros 'h' elementos
    unsigned int *limit = &shared_mem[h];
    if (threadIdx.x < h) {
        hlsh[threadIdx.x] = shg[threadIdx.x];
        hlsh[threadIdx.x] += psv[threadIdx.x + blockIdx.x * h];
    }
    unsigned int tamFaixa = (nMax - nMin + h) / h; //Calcula o teto do numero de possiveis valores sobre o numero de faixas
    //Calcula o limite superior do intervalo de valores de cada faixa e guarda no vetor limit
    if (threadIdx.x < h)
        limit[threadIdx.x] = nMin + tamFaixa * (threadIdx.x + 1);
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
                    output[atomicAdd(hlsh+i, 1)] = aux;
                    break;
                }
            }
        }
        start += d;
    }
}


__device__ inline void Comparator(uint &keyA, uint &keyB, uint dir) {
  uint t;

  if ((keyA > keyB) == dir) {
    t = keyA;
    keyA = keyB;
    keyB = t;
  }
}




__global__ void bitonicSortShared(unsigned int *d_DstKey, unsigned int *d_SrcKey,
                                  unsigned int arrayLength, 
                                  unsigned int paddedLength,
                                  unsigned int dir) {
  
    extern __shared__ unsigned int s_key[];

    unsigned int idx1 = threadIdx.x;
    unsigned int idx2 = threadIdx.x + (paddedLength / 2);

    if (idx1 < arrayLength) {
        s_key[idx1] = d_SrcKey[idx1];
    } else {
        s_key[idx1] = UINT32_MAX;
    }
    if (idx2 < arrayLength) {
        s_key[idx2] = d_SrcKey[idx2];
    } else {
        s_key[idx2] = UINT32_MAX;
    }

    __syncthreads();

    for (unsigned int size = 2; size < paddedLength; size <<= 1) {
        // Bitonic merge
        unsigned int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos + 0], s_key[pos + stride], ddd);
        }
    }

    // ddd == dir for the last bitonic merge step
    for (unsigned int stride = paddedLength / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos + 0], s_key[pos + stride], dir);
    }

    __syncthreads();
    if (idx1 < arrayLength) {
        d_DstKey[idx1] = s_key[idx1];
    }
    if (idx2 < arrayLength) {
        d_DstKey[idx2] = s_key[idx2];
    }

}

unsigned int proximaPotenciaDe2_Limite1024(unsigned int n) {
  if (n <= 1)   return 1;
  if (n <= 2)   return 2;
  if (n <= 4)   return 4;
  if (n <= 8)   return 8;
  if (n <= 16)  return 16;
  if (n <= 32)  return 32;
  if (n <= 64)  return 64;
  if (n <= 128) return 128;
  if (n <= 256) return 256;
  if (n <= 512) return 512;
  return 1024; // Se for <= 1024
}

bool verifySort(unsigned int* h_mppSorted, unsigned int* h_thrustSorted, long long n) {
    for(long long i = 0; i < n; i++) {
        if(h_mppSorted[i] != h_thrustSorted[i]) {
            fprintf(stderr, "ERRO de Verificacao no indice %lld!\n", i);
            fprintf(stderr, "  Esperado (thrust): %u\n", h_thrustSorted[i]);
            fprintf(stderr, "  Obtido (mppSort): %u\n", h_mppSorted[i]);
            // Imprime alguns valores ao redor do erro
            for (long long j = std::max(0LL, i - 5); j < std::min(n, i + 5); j++) {
                fprintf(stderr, "    idx %lld: mpp=%u, thrust=%u %s\n",
                    j, h_mppSorted[j], h_thrustSorted[j], (j==i) ? "<- ERRO" : "");
            }
            return false;
        }
    }
    return true;
}

void printCorrectUse() {
    printf("Uso correto eh:\n"
           "./mppSort <nTotalElements> <h> <hr>\n");
}

int main (int argc, char** argv) {
    if (argc != 4) {
        printCorrectUse();
        exit(1);
    }

    long long nTotal = atoll(argv[1]);
    long long h = atoll(argv[2]);
    int nR = atoi(argv[3]);

    // Por conta do tamanho dos histogramas
    if (h > SHARED_SIZE_LIMIT) {
        printf("h não pode ser maior que %d\n", SHARED_SIZE_LIMIT);
        exit(1);
    }

    printf("Configuracao do Teste:\n");
    printf(" - Numero de Repeticoes (nR): %d\n", nR);
    printf(" - Tamanho do Buffer: %lld elementos \n", nTotal);
    printf(" - Numero de Particoes (h): %lld\n", h);
    printf("--------------------------------------------------------\n");

    //Vetor principal alocado
    std::vector<unsigned int> input(nTotal);
    std::vector<unsigned int> outputThrust(nTotal);
    std::vector<unsigned int> outputKernel(nTotal);


    srand(time(NULL));
    unsigned int nMax;
    unsigned int nMin;
    for (size_t i = 0; i < nTotal; i++) {
        unsigned int a = rand();
        unsigned int b = rand();

        unsigned int v = a;
        if (i == 0) {
            nMax = v;
            nMin = v;
        } else if (v < nMin) {
            nMin = v;
        } else if (v > nMax) {
            nMax = v;
        }
        input[i] = v;
    }
    unsigned int L = (nMax - nMin + h) / h;


    printf("Intervalo de dados [nMin, nMax]: [%u, %u]\n", nMin, nMax);
    printf("Largura da Faixa (L): %u\n", L);
    printf("--------------------------------------------------------\n");
    

        // --- 2. Alocação de Memória (Device) ---
    unsigned int *d_Input, *d_Output, *d_Thrust;
    unsigned int *d_HH, *d_Hg, *d_SHg, *d_PSv;

    // Parâmetros dos kernels
    // Kernel 1 & 4
    int nt = 1024; // Threads por bloco
    int nb = 128;  // Número de bloco



    printf("Alocando memoria na GPU...\n");
    CUDA_CHECK(cudaMalloc(&d_Input, nTotal * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Output, nTotal * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Thrust, nTotal * sizeof(int)));
    
    // --- 3. Cópia Host -> Device ---
    printf("Copiando dados para a GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_Input, input.data(), nTotal * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Thrust, input.data(), nTotal * sizeof(unsigned int), cudaMemcpyHostToDevice));

    
    // Matriz HH: nb linhas, h colunas
    CUDA_CHECK(cudaMalloc(&d_HH, nb * h * sizeof(unsigned int)));
    // Vetor Hg: h elementos
    CUDA_CHECK(cudaMalloc(&d_Hg, h * sizeof(unsigned int)));
    // Vetor SHg: h elementos
    CUDA_CHECK(cudaMalloc(&d_SHg, h * sizeof(unsigned int)));
    // Matriz PSv: nb linhas, h colunas
    CUDA_CHECK(cudaMalloc(&d_PSv, nb * h * sizeof(unsigned int)));

    // --- 4. Execução dos Kernels ---
    printf("Executando kernels (%d repeticoes)...\n", nR);

    double totalTime1 = 0.0;

    size_t sharedMemK1 = 2 * h * sizeof(unsigned int);

    for (int i = 0; i < nR; i++) {
        CUDA_CHECK(cudaMemset(d_HH, 0, nb * h * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_Hg, 0, h * sizeof(unsigned int)));

        auto start = std::chrono::high_resolution_clock::now();
        
        blockAndGlobalHisto<<<nb, nt, sharedMemK1>>>(d_HH, d_Hg, h, d_Input, nTotal, nMin, nMax);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        totalTime1 += elapsed.count();
    }

    double avg1 = totalTime1/nR;

    double totalTime2 = 0.0;

    unsigned int pot  = proximaPotenciaDe2_Limite1024(h);
    size_t sharedMemK2 = pot * sizeof(unsigned int);

    for (int i = 0; i < nR; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        
        GlobalHistoScan<<<1, pot, sharedMemK2>>>(d_Hg, d_SHg, h, pot);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        totalTime2 += elapsed.count();
    }

    double avg2 = totalTime2/nR;


    double totalTime3 = 0.0;

    size_t sharedMemK3 = nb * sizeof(unsigned int);
    for (int i = 0; i < nR; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        
        verticalScanHH<<<h, nb, sharedMemK3>>>(d_HH, d_PSv, h, nb);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        totalTime3 += elapsed.count();
    }

    double avg3 = totalTime3/nR;

    double totalTime4 = 0.0;

    size_t sharedMemK4 = h * 2 * sizeof(unsigned int);
    for (int i = 0; i < nR; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        
        partitionKernel<<<nb,nt,sharedMemK4>>>( d_HH, d_SHg, d_PSv, h, d_Input, d_Output, nTotal, nMin, nMax ); 
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        totalTime4 += elapsed.count();
    }

    double avg4 = totalTime4/nR;


    std::vector<unsigned int> h_Hg(h);
    std::vector<unsigned int> h_SHg(h);
    CUDA_CHECK(cudaMemcpy(h_Hg.data(), d_Hg, h * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_SHg.data(), d_SHg, h * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned int* d_Output_backup;
    CUDA_CHECK(cudaMalloc(&d_Output_backup, nTotal * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_Output_backup, d_Output, nTotal * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    double totalTime5 = 0.0;
    for (int r = 0; r < nR; r++) {
        CUDA_CHECK(cudaMemcpy(d_Output, d_Output_backup, nTotal * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < h; i++) {
            unsigned int bin_size = h_Hg[i];
            if (bin_size == 0) continue;
            unsigned int bin_start_idx = h_SHg[i];
            unsigned int *d_Src_i = d_Output  + bin_start_idx;

            if (bin_size > 0 && bin_size <= SHARED_SIZE_LIMIT) {

                pot = proximaPotenciaDe2_Limite1024(bin_size);
                bitonicSortShared<<<1, pot/2, pot * sizeof(unsigned int)>>>(d_Src_i, d_Src_i, h_Hg[i], pot, 1);

            } else if (bin_size > 0) {
                // Para faixas maiores que a shared memory, usamos Thrust
                thrust::sort(thrust::device, d_Src_i, d_Src_i + bin_size);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        totalTime5 += elapsed.count();
    }
    double avg5 = totalTime5/nR;

    double tempoTotal = avg1 + avg2 + avg3 + avg4 + avg5;


    double totalTime_thrust_accum = 0.0;
    for(int r = 0; r < nR; r++) {
        CUDA_CHECK(cudaMemcpy(d_Thrust, input.data(), nTotal * sizeof(unsigned int), cudaMemcpyHostToDevice));

        auto start_thrust = std::chrono::high_resolution_clock::now();
        
        thrust::sort(thrust::device, d_Thrust, d_Thrust + nTotal);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end_thrust = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_thrust_run = end_thrust - start_thrust;
        totalTime_thrust_accum += elapsed_thrust_run.count();
    }
    double avg_thrust = totalTime_thrust_accum / nR;


    CUDA_CHECK(cudaMemcpy(outputThrust.data(), d_Thrust, nTotal * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputKernel.data(), d_Output, nTotal * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    if (verifySort(outputKernel.data(), outputThrust.data(), nTotal)) {
        printf("O vetor foi ordenado corretamente\n");
    } else {
        printf("O vetor foi ordenado incorretamente\n");
    }
    

    printf("Tempo blockAndGlobalHisto: %.4fs\n", avg1/1000);
    printf("Tempo GlobalHistoScan: %.4fs\n", avg2/1000);
    printf("Tempo verticalScanHH: %.4fs\n", avg3/1000);
    printf("Tempo partitionKernel: %.4fs\n", avg4/1000);
    printf("Tempo ordenacao: %.4fs\n", avg5/1000);
    printf("Tempo total: %.4fs\n", tempoTotal/1000);
    printf("Tempo thrust: %.4fs\n", avg_thrust/1000);
    printf("Vazão mpp: %.4fGE/s\n", (nTotal/1000000.0)/tempoTotal);
    printf("Vazão thrust: %.4fGE/s\n", (nTotal/1000000.0)/avg_thrust);
    printf("Speedup: %.4f\n", avg_thrust/tempoTotal);

}