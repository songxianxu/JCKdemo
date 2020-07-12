# Some collections of sample CUDA kernel written in Julia

This repo collects some of the CUDA kernel functions written in Julia using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). The samples mainly follows 
    
- [CUDA by Example](https://developer.nvidia.com/cuda-example)
- [CUDA sample](https://github.com/NVIDIA/cuda-samples)

## Basics

Check properties of graphics cards 

- I am not sure of the best equivalence in Julia.
- `enum_gpu.cu` from CUDA by Example that uses `cudaDeviceProp`.
- `deviceQuery.cu` from CUDA samples. This comes along with CUDA installation. e.g., `/opt/cuda-10.1/samples/`
- `deviceQuery` from demo. Run `/opt/cuda-10.1/extras/demo_suite/deviceQuery` and the GPU info will be listed. e.g.,

```    
/opt/cuda-10.1/extras/demo_suite/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 1660"
  CUDA Driver Version / Runtime Version          10.1 / 10.1
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 5912 MBytes (6198788096 bytes)
  (22) Multiprocessors, ( 64) CUDA Cores/MP:     1408 CUDA Cores
  GPU Max Clock rate:                            1830 MHz (1.83 GHz)
  Memory Clock rate:                             4001 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 1572864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.1, CUDA Runtime Version = 10.1, NumDevs = 1, Device0 = GeForce GTX 1660
Result = PASS
```

Memory Allocation   

Memory Free   


## CUDA by Example
    
- Chapter04 
    - add_loop_cpu.cu -> add_loop_cpu.jl
    - add_loop_gpu.cu -> add_loop_gpu.jl
- Chapter05
    - add_loop_threads.cu -> add_loop_threads.jl
    - add_loop_long_blocks.cu -> add_loop_long_blocks.jl
    - dot.cu -> dot.jl 


## CUDA Samples

- matrixMul
- matrixMulCUBLAS





