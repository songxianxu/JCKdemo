#= Codes from https://juliagpu.gitlab.io/CUDA.jl/tutorials/introduction/ =#
using CUDA
using BenchmarkTools
using Test

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



