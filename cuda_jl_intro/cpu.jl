# Code from https://juliagpu.gitlab.io/CUDA.jl/tutorials/introduction/

using BenchmarkTools

N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

y .+= x             # increment each element of y with the corresponding element of x


# Sequential addition CPU
function sequential_add!(y, x)
  for i in eachindex(y, x)
    @inbounds y[i] += x[i]
  end
  return nothing
end

fill!(y, 2)
#= sequential_add!(y, x) =#


# Parallel addition CPU
function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
#= parallel_add!(y, x) =#



@show @btime sequential_add!(y, x)
@show @btime parallel_add!(y, x)
# Threads reset 
# Threads.nthreads() = 4 
