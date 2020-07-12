#= Codes from https://juliagpu.gitlab.io/CUDA.jl/tutorials/introduction/ =#
using CUDA
using BenchmarkTools
using Test

N = 2^20
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

@show @btime add_broadcast!($y_d, $x_d)

## CUDA.@sync
# Block CPU until the queued GPU tasks are done.
# Similar to Base.@sync waits for distributed CPU tasks.

## GPU addition kernel
function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function gpu_add2!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end



fill!(y_d, 2)
println("================================")
@show @time @cuda gpu_add1!(y_d, x_d)
# 0.298084s
println("================================")
@show @time gpu_add1!(y_d, x_d)
# 13.997580s # With warning about scalar operations
println("================================")
@show @time @cuda gpu_add1!(y_d, x_d)
# 0.000098s
println("================================")
@show @time gpu_add2!(y_d,x_d) 
# 13.794801s
# It seems the addition is done by CPU, each time the data is copied back to CPU
println("================================")

## !!! @cuda macro
# It will compile the kernel (gpu_add1!) for execution on the GPU. Once compiled, future invocations are fast. 


## Benchmark
function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

@show @btime bench_gpu1!($y_d, $x_d)
# 53.233ms (46 allocations: 1.22kiB)

# Check the consequence if we do not sync
@show @btime @cuda gpu_add1!($y_d, $x_d)
# 11.069μs (56 allocations: 1.70KiB)
# This is not the actual time for GPU computation.


# Parallel GPU kernel
function gpu_add3!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads=256 gpu_add3!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)


function bench_gpu3!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add3!(y, x)
    end
end

@btime bench_gpu3!($y_d, $x_d)
#1.283ms
