#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void sumArraysOnGpu(float *A, float *B, float *C, int fatorUnroll) {
  unsigned int idx = blockIdx.x * blockDim.x * fatorUnroll + threadIdx.x;

  for (int i = 1; i <= fatorUnroll; i++) {
    int index = idx + fatorUnroll;
    C[index] = A[index] + B[index];
  }
}

void initialData(float *ip, int size){
  // generate different seed for random number
  time_t t;
  srand((unsigned int) time (&t) - ip[0]);

  for (int i=0; i<size; i++){
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

void linearData(float *input, int size) {
  for (int i = 0; i < size; i++) {
    input[i] = i + (size / (1024 * 1e3));
  }
}

int main(int argc, char **argv){
  int expoente = atoi(argv[1]); // Primeiro argumento é o expoente onde 2^X = tamanho do elemento
  int threads = atoi(argv[2]); // Segundo argumento é o numero de threads
  int fatorUnroll = atoi(argv[3]); // Terceiro argumento é o fator de unroll

  size_t nBytes = (2 << (expoente + 1)) / sizeof(float);
  int nElem = nBytes / sizeof(float);

  float *h_A, *h_B, *h_C;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);

  initialData(h_A, nElem);
  linearData(h_B, nElem);

  printf("Quantidade de elementos: %d \n Quantidade de MB: %lu MB\n\n", nElem, (nBytes / (1024*1024)));

  float *d_A, *d_B, *d_C;
  cudaMalloc((float**)&d_A, nBytes);
  cudaMalloc((float**)&d_B, nBytes);
  cudaMalloc((float**)&d_C, nBytes);

  // Use cudaMemcpy to transfer the data from the host memory to the GPU global memory with the
  // parameter cudaMemcpyHostToDevice specifying the transfer direction.
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

  sumArraysOnGpu<<<(nElem / fatorUnroll) / threads, threads>>>(d_A, d_B, d_C, fatorUnroll);

  cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

  free(h_A);
  free(h_B);
  free(h_C);

  // use cudaFree to release the memory used on the GPU
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaDeviceReset();

  return (0);
}
