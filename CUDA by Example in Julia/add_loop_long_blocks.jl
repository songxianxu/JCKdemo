using CUDA
using Test

function add!(a, b, c)
  tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  while (tid <= length(a)) 
    c[tid] = a[tid] + b[tid]
    tid += blockDim().x * gridDim().x

  end
  return 
end

#= Max threads per block = 1024 =# 
#= Max threads dimensions = 1024 1024 64 =#

const N = 32 * 1024

a = Vector{Int}(undef,N)
b = Vector{Int}(undef,N)
c = Vector{Int}(undef,N)

for i = 0:N-1
  a[i+1] = -i 
  b[i+1] = i^2
end
c = a + b

dev_a = cu(a)
dev_b = cu(b)
dev_c = cu(c)
#= copyto! =#

@cuda blocks=128 threads=128 add!(dev_a ,dev_b ,dev_c)
# add<<<128,128>>>(dev_a, dev_b, dev_c)
@test Array(dev_c) == c
