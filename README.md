#Bibliografia
https://helpmanual.io/help/nvprof/
http://www.hds.bme.hu/~fhegedus/C++/Professional%20CUDA%20C%20Programming.pdf
https://github.com/andersy005/cuda-programming/tree/master/cuda-c/src


# Atividade 1:
1. Modificar o ​​exemplo​ ​para ​​executar ​1​ ​bloco​​ com​​ 1024​​ threads.​​ Imprimir​​ e​​ verificar ​​a ordem​ ​de​ ​execução​ ​dos​ ​threads​.​ ​É​ ​sequencial​ ​?​ ​É​ ​determinística​ ​?​ ​É​ ​agrupada​ ​? (output0.txt)

> É agrupada em grupos de 32, sequencialmente. Porém a ordem de execução dos grupos não são essencialmente sequenciais.

2. Agora ​​executar ​​1024 ​​blocos ​​com ​​1 ​​thread ​​cada. ​​Imprimir ​ ​e​​verificar​​ a​ ordem​​ de execução​ ​dos​
​threads​. É​ ​sequencial​ ​?​ ​É​ ​determinística​ ​?​ ​É​ ​agrupada​ ​? (output1.txt)

> Tem que analisar o output1.txt....

3. Compilar ​​com ​​capability ​​3.5 ​e ​​gerar ​​2 ​​Trilhões ​​de ​​threads ​​(bloco​\*​​thread),​​ maximo que​ ​a​ ​GPU​ ​poderá.​ ​Usar​ ​if​ ​para​ ​imprimir​ ​apenas​ ​alguns.

> Não funciona (ainda)

4. Executar​​ alguma​​ operação​​ na​​ GPU.

> Executado milhões de cálculos de fibonacci


# Atividade 2:
```
Tamanho dos vetores: 1.024
==14123== NVPROF is profiling process 14123, command: ./ex1
==14123== Profiling application: ./ex1
==14123== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   45.95%  4.3530us         2  2.1760us  2.1130us  2.2400us  [CUDA memcpy HtoD]
                   34.79%  3.2960us         1  3.2960us  3.2960us  3.2960us  [CUDA memcpy DtoH]
                   19.25%  1.8240us         1  1.8240us  1.8240us  1.8240us  sumArraysOnGpu(float*, float*, float*, int)
      API calls:   68.52%  58.264ms         3  19.421ms  3.1830us  58.257ms  cudaMalloc
                   30.08%  25.582ms         1  25.582ms  25.582ms  25.582ms  cudaDeviceReset
                    0.89%  758.65us        96  7.9020us     298ns  336.60us  cuDeviceGetAttribute
                    0.18%  151.33us         3  50.442us  4.5520us  138.71us  cudaFree
                    0.12%  99.297us         1  99.297us  99.297us  99.297us  cuDeviceGetName
                    0.10%  85.648us         1  85.648us  85.648us  85.648us  cuDeviceTotalMem
                    0.07%  63.420us         3  21.140us  16.268us  30.868us  cudaMemcpy
                    0.02%  21.076us         1  21.076us  21.076us  21.076us  cudaLaunchKernel
                    0.00%  3.2340us         3  1.0780us     369ns  1.9670us  cuDeviceGetCount
                    0.00%  2.2420us         1  2.2420us  2.2420us  2.2420us  cuDeviceGetPCIBusId
                    0.00%  1.1540us         2     577ns     442ns     712ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetUuid

Tamanho dos vetores: 65.536
==14449== NVPROF is profiling process 14449, command: ./ex1
==14449== Profiling application: ./ex1
==14449== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.55%  353.20us         2  176.60us  176.18us  177.01us  [CUDA memcpy HtoD]
                   30.26%  155.89us         1  155.89us  155.89us  155.89us  [CUDA memcpy DtoH]
                    1.19%  6.1130us         1  6.1130us  6.1130us  6.1130us  sumArraysOnGpu(float*, float*, float*, int)
      API calls:   68.16%  58.345ms         3  19.448ms  3.3580us  58.337ms  cudaMalloc
                   29.58%  25.321ms         1  25.321ms  25.321ms  25.321ms  cudaDeviceReset
                    0.98%  837.53us        96  8.7240us     298ns  375.20us  cuDeviceGetAttribute
                    0.84%  718.77us         3  239.59us  48.040us  442.85us  cudaMemcpy
                    0.18%  151.56us         3  50.520us  4.9060us  141.04us  cudaFree
                    0.12%  101.09us         1  101.09us  101.09us  101.09us  cuDeviceGetName
                    0.11%  94.320us         1  94.320us  94.320us  94.320us  cuDeviceTotalMem
                    0.03%  21.732us         1  21.732us  21.732us  21.732us  cudaLaunchKernel
                    0.00%  2.7730us         1  2.7730us  2.7730us  2.7730us  cuDeviceGetPCIBusId
                    0.00%  2.6220us         3     874ns     362ns  1.6460us  cuDeviceGetCount
                    0.00%     986ns         2     493ns     336ns     650ns  cuDeviceGet
                    0.00%     401ns         1     401ns     401ns     401ns  cuDeviceGetUuid

Tamanho dos vetores: 16.777.216
==13290== NVPROF is profiling process 13290, command: ./ex1
==13290== Profiling application: ./ex1
==13290== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.21%  89.294ms         2  44.647ms  44.586ms  44.708ms  [CUDA memcpy HtoD]
                   30.92%  40.474ms         1  40.474ms  40.474ms  40.474ms  [CUDA memcpy DtoH]
                    0.88%  1.1510ms         1  1.1510ms  1.1510ms  1.1510ms  sumArraysOnGpu(float*, float*, float*, int)
      API calls:   60.54%  131.51ms         3  43.836ms  43.253ms  44.787ms  cudaMemcpy
                   26.94%  58.529ms         3  19.510ms  237.84us  58.027ms  cudaMalloc
                   11.81%  25.644ms         1  25.644ms  25.644ms  25.644ms  cudaDeviceReset
                    0.35%  760.22us        96  7.9180us     296ns  346.70us  cuDeviceGetAttribute
                    0.26%  562.76us         3  187.59us  169.61us  216.97us  cudaFree
                    0.04%  95.749us         1  95.749us  95.749us  95.749us  cuDeviceGetName
                    0.04%  84.091us         1  84.091us  84.091us  84.091us  cuDeviceTotalMem
                    0.01%  31.774us         1  31.774us  31.774us  31.774us  cudaLaunchKernel
                    0.00%  2.8540us         1  2.8540us  2.8540us  2.8540us  cuDeviceGetPCIBusId
                    0.00%  2.7690us         3     923ns     317ns  1.7230us  cuDeviceGetCount
                    0.00%  1.1550us         2     577ns     398ns     757ns  cuDeviceGet
                    0.00%     401ns         1     401ns     401ns     401ns  cuDeviceGetUuid
```

## Solicitados:
### 100M
```
Tamanho dos vetores: 100.000.000
==15333== NVPROF is profiling process 15333, command: ./ex1
==15333== Profiling application: ./ex1
==15333== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.14%  528.80ms         2  264.40ms  264.28ms  264.52ms  [CUDA memcpy HtoD]
                   30.99%  240.47ms         1  240.47ms  240.47ms  240.47ms  [CUDA memcpy DtoH]
                    0.87%  6.7471ms         1  6.7471ms  6.7471ms  6.7471ms  sumArraysOnGpu(float*, float*, float*, int)
      API calls:   89.83%  776.49ms         3  258.83ms  248.38ms  264.63ms  cudaMemcpy
                    6.95%  60.078ms         3  20.026ms  618.33us  58.831ms  cudaMalloc
                    2.96%  25.577ms         1  25.577ms  25.577ms  25.577ms  cudaDeviceReset
                    0.14%  1.2313ms         3  410.43us  392.85us  444.13us  cudaFree
                    0.09%  761.83us        96  7.9350us     297ns  349.48us  cuDeviceGetAttribute
                    0.01%  100.23us         1  100.23us  100.23us  100.23us  cuDeviceGetName
                    0.01%  84.679us         1  84.679us  84.679us  84.679us  cuDeviceTotalMem
                    0.00%  30.254us         1  30.254us  30.254us  30.254us  cudaLaunchKernel
                    0.00%  2.8610us         3     953ns     335ns  1.6700us  cuDeviceGetCount
                    0.00%  2.4150us         1  2.4150us  2.4150us  2.4150us  cuDeviceGetPCIBusId
                    0.00%  1.1440us         2     572ns     402ns     742ns  cuDeviceGet
                    0.00%     430ns         1     430ns     430ns     430ns  cuDeviceGetUuid
.
Tamanho dos vetores: 133000000
==15587== NVPROF is profiling process 15587, command: ./ex1
==15587== Profiling application: ./ex1
==15587== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.21%  704.11ms         2  352.05ms  351.56ms  352.55ms  [CUDA memcpy HtoD]
                   30.93%  319.27ms         1  319.27ms  319.27ms  319.27ms  [CUDA memcpy DtoH]
                    0.86%  8.9286ms         1  8.9286ms  8.9286ms  8.9286ms  sumArraysOnGpu(float*, float*, float*, int)
      API calls:   92.11%  1.03283s         3  344.28ms  329.32ms  351.83ms  cudaMemcpy
                    5.38%  60.332ms         3  20.111ms  752.57us  58.826ms  cudaMalloc
                    2.29%  25.721ms         1  25.721ms  25.721ms  25.721ms  cudaDeviceReset
                    0.14%  1.5158ms         3  505.27us  477.87us  530.08us  cudaFree
                    0.06%  714.02us        96  7.4370us     298ns  324.79us  cuDeviceGetAttribute
                    0.01%  104.74us         1  104.74us  104.74us  104.74us  cuDeviceGetName
                    0.01%  84.997us         1  84.997us  84.997us  84.997us  cuDeviceTotalMem
                    0.00%  32.284us         1  32.284us  32.284us  32.284us  cudaLaunchKernel
                    0.00%  2.4200us         3     806ns     335ns  1.5440us  cuDeviceGetCount
                    0.00%  2.4070us         1  2.4070us  2.4070us  2.4070us  cuDeviceGetPCIBusId
                    0.00%     993ns         2     496ns     320ns     673ns  cuDeviceGet
                    0.00%     401ns         1     401ns     401ns     401ns  cuDeviceGetUuid
.

Não rodou pra valores maiores, nem de perto pra 200M ou 300M
Out of memory.
```
