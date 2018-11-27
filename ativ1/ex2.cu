#include <stdlib.h>
#include <stdio.h>

__global__ void helloFromGPU() {
  printf("Hello World from GPU! %d %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char**argv) {
  printf("Hello World from CPU!\n");

  long long int threads = 2 * 1024 * 1024;
  long long int blocks = 1024 * 1024;
  helloFromGPU<<<threads, blocks>>>();

  cudaDeviceReset();
  // cudaDeviceSynchronize();
  // CHECK(cudaDeviceReset());

  return 0;
}
