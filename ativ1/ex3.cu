#include <stdio.h>

__global__ void helloFromGPU() {
  if (threadIdx.x < 20 && blockIdx.x < 20)
    printf("Hello World from GPU! %d %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char**argv) {
  printf("Hello World from CPU!\n");

  // Não tá funcionando colocar esse numero bem grande...
  long long int blocks = 1024;
  long long int threads = 2;
  helloFromGPU<<<blocks, threads>>>();

  cudaDeviceReset();
  return 0;
}
