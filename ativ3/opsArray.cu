#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumArraysOnGpu(float *A, float *B, float *C){
  int idx = threadIdx.x;
  C[idx] = A[idx] + B[idx];
}

__global__ void mathOperationsOnGPU(float *A, float *B, float *C, int operations) {
  int idx = threadIdx.x;
  float result;

  for (int i = 0; i < operations; i++) {
    int r = i % 6;

    switch (r) {
      case 0:
        result += B[idx];
        break;
      case 1:
      result -= A[idx];
        break;
      case 2:
        result += 9;
        break;
      case 3:
        result -= 9203.34;
        break;
      case 4:
        result *= 0.2;
        break;
      case 5:
        result -= (A[idx] / 1024);
        break;
    }
  }

  C[idx] = result;
}


void initialData(float *ip, int size){
  // generate different seed for random number
  time_t t;
  srand((unsigned int) time (&t));

  for (int i=0; i<size; i++){
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

void linearData(float *input, int size) {
  for (int i = 0; i < size; i++) {
    input[i] = i + (size / (1024 * 1e3));
  }
}

int main(int argc, char **argv) {
  int expoente = atoi(argv[1]); // Primeiro argumento é o expoente onde 2^X = tamanho do elemento
  int blocks = atoi(argv[2]); // Primeiro argumento é a quantidade de blocos
  int operations = atoi(argv[3]); // Segundo argumento é a quantidade de operações matemáticas por thread

  srand(time(NULL));

  size_t nBytes = (2 << (expoente + 1)) / sizeof(float);
  int nElem = nBytes / sizeof(float);

  float *h_A, *h_B, *h_C, *result;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);
  result = (float *)malloc(nBytes);

  initialData(h_A, nElem);
  linearData(h_B, nElem);

  printf("Quantidade de elementos: %d \n Quantidade de MB: %lu MB, Quantidade de operações: %d\n\n", nElem, (nBytes / (1024*1024)), operations);

  float *d_A, *d_B, *d_C;
  cudaMalloc((float**)&d_A, nBytes);
  cudaMalloc((float**)&d_B, nBytes);
  cudaMalloc((float**)&d_C, nBytes);

  // Use cudaMemcpy to transfer the data from the host memory to the GPU global memory with the
  // parameter cudaMemcpyHostToDevice specifying the transfer direction.
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

  mathOperationsOnGPU<<<blocks, 1024>>>(d_A, d_B, d_C, operations);
  // sumArraysOnGpu<<<1, nElem>>>(d_A, d_B, d_C);
  cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

  free(h_A);
  free(h_B);
  free(h_C);
  free(result);

  // use cudaFree to release the memory used on the GPU
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaDeviceReset();

  return (0);
}
