using Statistics
using Flux
using BFloat16s

struct LN
    γ
    β
    ϵ
    dims
end
LN(γ, β) = LN(γ, β, 1e-5, 1)
LN(n::Integer) = LN(ones(Float32, n), zeros(Float32, n))
@Flux.functor LN
Flux.trainable(m::LN) = (γ=m.γ, β=m.β)
function (m::LN)(x::AbstractArray{T,N}) where {T,N}
    μ = mean(x, dims=m.dims)
    σ² = var(x, dims=m.dims)
    (T).(m.γ .* (x .- μ) ./ (σ² .+ m.ϵ).^0.5 .+ m.β)
end

bf16(m) = Flux._paramtype(BFloat16, m)