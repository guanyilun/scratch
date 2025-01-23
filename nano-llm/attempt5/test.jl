using MLUtils
using Flux

a = collect(0:9) .|> Float32
r = repeat([0.5], 10)
data = map((a, r) -> (a, r), a, r)

using ThreadedScans

f = (left, right) -> begin
    v_left, r_left = left
    v_right, r_right = right
    v_left * r_right + v_right, r_left * r_right
end

ThreadedScans.scan!(f, data)
data