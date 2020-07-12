using CUDA
using BenchmarkTools

N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

@cuda threads=256 gpu_add2!(y_d, x_d)

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=1408 gpu_add2!(y, x)
    end
end
# Here the threads corresponds to the CUDA cores of the hardware
# For GTX 1660, it has 1408 cuda cores.
# 22 SMs 

@btime bench_gpu2!($y_d, $x_d)
