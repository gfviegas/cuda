#include <stdio.h>

__global__ void helloFromGPU() {
  printf("Hello World from GPU! %d %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char**argv) {
  printf("Hello World from CPU!\n");

  int blocks = 1024;
  int threads = 1;
  helloFromGPU<<<blocks, threads>>>();

  cudaDeviceReset();
  return 0;
}
