using Flux
using Statistics
using MLUtils

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

time_mix(x, x_prev, mix) = @. x * mix + x_prev * (1 - mix)
square_relu(x) = max(0, x)^2

mutable struct State{T <: AbstractArray{<:Number, 2}}
    x_tm::T  # token mixing
    x_cm::T  # channel mixing
    a::T
    b::T
    p::T
end
State(k, v; n_layer::Integer) = begin
    @assert size(k) == (length(k),)
    target_dim = (size(v)[1], n_layer)
    State(zeros_like(v, target_dim), zeros_like(v, target_dim), repeat(v, 1, n_layer), ones_like(v, target_dim), repeat(k, 1, n_layer))
end


function wkv!(state::State, k, v; i, u, w)
    a, b, p = @views state.a[:,i], state.b[:,i], state.p[:,i]

    # get y from a, b, p
    q = @. max(p, u+k)
    c = @. exp(p - q)*a + exp(u+k-q)*v
    d = @. exp(p - q)*b + exp(u+k-q)
    wkv = @. c/d

    # update states
    q = @. max(p+w, k)
    a = @. exp(p+w-q)*a + exp(k-q)*v
    b = @. exp(p+w-q)*b + exp(k-q)
    p = @. q

    @views state.a[:,i] .= a 
    @views state.b[:,i] .= b 
    @views state.p[:,i] .= p

    # these two can be written using a same function but
    # for readability I keep them separate. It directly
    # follows the table here (last column):
    # https://github.com/BlinkDL/RWKV-LM#rwkv-4-improvements

    wkv, state
end

struct TokenMixing{T}
    Tₖ::AbstractArray{T, 1}
    Tᵥ::AbstractArray{T, 1}
    Tᵣ::AbstractArray{T, 1}
    r_proj
    k_proj
    v_proj
    out_proj
    time_first::AbstractArray{T, 1}
    time_decay::AbstractArray{T, 1}
    n_layer::Integer
end
@Flux.functor TokenMixing
TokenMixing(T, n_embed::Integer, n_layer::Integer) = TokenMixing(
    zeros(T, n_embed), # Tₖ
    zeros(T, n_embed), # Tᵥ
    zeros(T, n_embed), # Tᵣ
    Dense(n_embed, n_embed, bias=false), # r_proj
    Dense(n_embed, n_embed, bias=false), # k_proj
    Dense(n_embed, n_embed, bias=false), # v_proj
    Dense(n_embed, n_embed, bias=false), # out_proj
    zeros(T, n_embed), # time_first
    ones(T, n_embed),  # time_decay
    n_layer,
)

function (m::TokenMixing)(x, state; i)  # modifies state!
    if isnothing(state) 
        x_prev = zero(x)
    else
        @views x_prev = state.x_tm[:,i]
    end

    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵥ = time_mix(x, x_prev, m.Tᵥ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ)
    v = m.v_proj(xᵥ)


    if isnothing(state); state = State(k, v; n_layer=m.n_layer); end
    @views state.x_tm[:, i] .= x

    wkv, state = wkv!(state, k, v; i=i, u=m.time_first, w=m.time_decay)  # modify state in place!
    rwkv = m.out_proj(r .* wkv)

    rwkv, state
end

struct ChannelMixing{T}
    Tₖ::AbstractArray{T, 1}
    Tᵣ::AbstractArray{T, 1}
    r_proj
    k_proj
    v_proj
end
@Flux.functor ChannelMixing
ChannelMixing(T, n_embed::Integer) = ChannelMixing(
    zeros(T, n_embed), # Tₖ
    zeros(T, n_embed), # Tᵣ
    Dense(n_embed, n_embed, bias=false), # r_proj
    Dense(n_embed, n_embed, bias=false), # k_proj
    Dense(n_embed, n_embed, bias=false), # v_proj
)

function (m::ChannelMixing)(x, state; i)
    xₖ = @views time_mix(x, state.x_cm[:,i], m.Tₖ)
    xᵣ = @views time_mix(x, state.x_cm[:,i], m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ) .|> square_relu

    @views state.x_cm[:, i] .= x

    r .* (m.v_proj(k)), state
end

struct Block
    ln1
    token_mixing
    ln2
    channel_mixing
end
@Flux.functor Block
Block(n_embed::Integer) = Block(
    LN(n_embed),
    TokenMixing(Float32, n_embed),
    LN(n_embed),
    ChannelMixing(Float32, n_embed),
)
(m::Block)(x, state; i) = begin
    xp, state = m.token_mixing(m.ln1(x), state; i=i)
    x = x + xp
    xp, state = m.channel_mixing(m.ln2(x), state; i=i)
    x = x + xp 
    x, state
end

struct RWKV
    ln_init
    embedding
    blocks
    ln_final
    lm_head
end
@Flux.functor RWKV
RWKV(n_embed::Integer, n_blocks::Integer, n_vocab::Integer) = RWKV(
    Embedding(n_vocab, n_embed),
    LN(n_embed),
    [Block(n_embed) for _ in 1:n_blocks],
    LN(n_embed),
    Embedding(n_embed, n_vocab)
)
(m::RWKV)(x, state=nothing) = begin
    x = m.embedding(x)
    x = m.ln_init(x)
    for i in 1:length(m.blocks)
        x, state = m.blocks[i](x, state; i=i)
    end
    x = m.ln_final(x)
    x = m.lm_head.weight' * x
    x, state
end