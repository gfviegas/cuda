#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

#define CHECK(call)                                         \
{                                                           \
  const cudaError_t error = call;                           \
  if (error != cudaSuccess) {                               \
    fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);  \
    fprintf(stderr, "code: %d, reason: %s\n", error,        \
    cudaGetErrorString(error));                             \
    exit(1);                                                \
  }                                                         \
}

void initialData(float *ip, int size){
  time_t t;
  srand((unsigned int) time (&t));

  for (int i = 0; i < size; i++){
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
  for (int idx = 0; idx < N; idx++) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void sumArraysOnGpu(float *A, float *B, float *C, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    C[i] = A[i] + B[i];
  }
}


int main(int argc, char**argv) {
  // Configura tamanho dos vetores
  int nElem = 177 * 1.e6;
  printf("Tamanho dos vetores: %d\n", nElem);

  // Alocando memoria na CPU
  size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *) malloc(nBytes);
  h_B = (float *) malloc(nBytes);
  hostRef = (float *) malloc(nBytes);
  gpuRef = (float *) malloc(nBytes);

  initialData(h_A, nElem);
  initialData(h_B, nElem);

  // Alocando memoria global (GPU)
  float *d_A, *d_B, *d_C;
  CHECK(cudaMalloc((float **)&d_A, nBytes));
  CHECK(cudaMalloc((float **)&d_B, nBytes));
  CHECK(cudaMalloc((float **)&d_C, nBytes));

  // Transferindo dados da CPU pra GPU
  CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
  // CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

  // Invocando o Kernel na CPU
  int iLen = 512;
  dim3 block(iLen);
  dim3 grid((nElem + block.x - 1) / block.x);
  sumArraysOnGpu<<<grid, block>>>(d_A, d_B, d_C, nElem);

  // Copia os resultados do Kernel de volta pra CPU
  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

  // Libera memoria da GPU
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C));

  // Libera memória da CPU
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  cudaDeviceReset();
  return 0;
}
