function add!(a ,b, c)
  tid = 1
  while tid <= length(a)
    c[tid] = a[tid] + b[tid]
    tid += 1
  end
end

a = Vector{Int}(undef,10)
b = Vector{Int}(undef,10)
c = Vector{Int}(undef,10)

for i = 0:9
  a[i+1] = -i 
  b[i+1] = i^2
end

add!(a ,b ,c)
@show a
@show b
@show c
