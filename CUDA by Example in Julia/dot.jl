using CUDA
using Test
import LinearAlgebra: dot

const threadsPerBlock = 256 
const N =  1024
imin(a,b) = a < b ? a : b


const blocksPerGrid = imin(32, (N+threadsPerBlock-1) ÷ threadsPerBlock)

function dot!(a, b, c)
  cache = @cuStaticSharedMem(Float64, threadsPerBlock)
  tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  cacheIndex = threadIdx().x
  temp = 0 
  while (tid <= N) 
    temp += a[tid] * b[tid]
    tid += blockDim().x * gridDim().x
  end

  #= set the cache values =#
  cache[cacheIndex] = temp  

  #= synchronize threads in this block =#
  sync_threads()  

  #= for reductions, threadsPerBlock must be a power of 2 =#
  i = blockDim().x ÷ 2
  while i ≠ 0 
    if cacheIndex ≤ i
      cache[cacheIndex] += cache[cacheIndex + i] 
    end
    sync_threads()
    i = i ÷ 2
  end
            
  if cacheIndex == 1 
    c[blockIdx().x] = cache[1]
  end

  return 
end


a = Vector{Int}(undef, N)
b = Vector{Int}(undef, N)
#= partial_c = Vector{Int}(undef, blocksPerGrid) =#

a_dev = CuArray{Int}(undef, N)
b_dev = CuArray{Int}(undef, N)
partial_c_dev = CuArray{Int}(undef, blocksPerGrid)

#= fill in the host memory with data =#
for i = 1:N
  a[i] = i-1 
  b[i] = (i-1)*2
end

copyto!(a_dev, a)
copyto!(b_dev, b)
#= Can copyto! do the inverse way? =#

@cuda blocks=blocksPerGrid threads=threadsPerBlock dot!(a_dev ,b_dev, partial_c_dev)
partial_c = Array(partial_c_dev)
c = sum(partial_c)

@test dot(a,b) == c
