### test alpha
using Wignerd

s1, s2 = 1, 2
f = x -> Wignerd.alpha(x, s1, s2)
f.(collect(5:9))

### test wignerd_init
costheta = Array([-0.75, -0.5 ,-0.25, 0.25, 0.5, 0.75])
wigd_hi = zero(costheta)
Wignerd.wigd_init!(s1, s2, costheta, wigd_hi)
wigd_hi

### test wignerd recur
wigd_lo = zero(costheta)
Wignerd.wigd_rec!(10, s1, s2, costheta, wigd_hi, wigd_lo)
wigd_hi

### test cf_from_cl
lmax = 10
cl = collect(0:lmax)
costheta = Array(range(-1, 1, length=10))
cf = Wignerd.cf_from_cl(s1, s2, lmax, cl, costheta)

### test cl_from_cf
weights = ones(length(costheta))
cl_recon = Wignerd.cl_from_cf(s1, s2, lmax, cf, costheta, weights)
