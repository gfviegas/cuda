#include <stdio.h>

__global__ void helloFromGPU() {
  if (threadIdx.x < 20 && blockIdx.x < 20)
    printf("Hello World from GPU! %d %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char**argv) {
  printf("Hello World from CPU!\n");

  // 2 milhões blocos de 1024 threads
  long long int blocks = 2 * 1e6;
  long long int threads = 1024; // Numero maximo suportada pela GPU que rodamos
  helloFromGPU<<<blocks, threads>>>();

  cudaDeviceReset();
  return 0;
}
