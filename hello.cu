// #include <stdio.h>

__global__ void helloFromGPU() {
  printf("Hello World from GPU! %d %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char**argv) {
  printf("Hello World from CPU!\n");
  helloFromGPU<<<4, 6>>>(); CHECK(cudaDeviceReset());

  return 0;
}
