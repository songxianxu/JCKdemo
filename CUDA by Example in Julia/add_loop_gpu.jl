using CUDA

function add!(a, b, c)
  tid = blockIdx().x
  if (tid <= length(a)) 
    c[tid] = a[tid] + b[tid]
  end
  return 
end

const N = 10000

a = Vector{Int}(undef,N)
b = Vector{Int}(undef,N)
c = Vector{Int}(undef,N)

for i = 0:N-1
  a[i+1] = -i 
  b[i+1] = i^2
end

dev_a = cu(a)
dev_b = cu(b)
dev_c = cu(c)

@cuda blocks=N add!(dev_a ,dev_b ,dev_c)
# add<<<N,1>>>(dev_a, dev_b, dev_c)

@show dev_a
@show dev_b
@show dev_c
