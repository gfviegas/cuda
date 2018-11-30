#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ void fibonacci(int n) {
  int novoN = abs((n - (int) blockIdx.x + (int) threadIdx.x) % n);

  int aux = novoN;
  long long int a = 0;
  long long int b = 1;

  while (aux-- > 1) {
    long long int t = a;
    a = b;
    b += t;
  }

  printf("Fibonacci de %d = %lld\n", novoN, b);
}

int main(int argc, char**argv) {
  printf("Hello World from CPU!\n");

  long long int threads = 1024;
  long long int blocks = 10;
  fibonacci<<<threads, blocks>>>(60);

  cudaDeviceSynchronize();
  // cudaDeviceReset();
  // CHECK(cudaDeviceReset());

  return 0;
}
