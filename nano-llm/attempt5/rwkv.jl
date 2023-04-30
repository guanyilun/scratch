using Flux
using MLUtils: batch, zeros_like
using NNlib: batched_mul

include("common.jl")

time_mix(x, x_prev, mix) = @. x * mix + x_prev * (1 - mix)
square_relu(x::AbstractFloat) = max(0, x)^2

mutable struct State
    x_tm  # token mixing
    x_cm  # channel mixing
    a
    b
    p # largest exponent seen
end

State(n_embed::Integer, n_layer::Integer) = begin
    dim = (n_embed, n_layer)
    # State(zeros(Float32, dim), zeros(Float32, dim), zeros(Float32, dim), zeros(Float32, dim))
    State(zeros(Float32, dim), zeros(Float32, dim), zeros(Float32, dim), zeros(Float32, dim), zeros(Float32, dim))
end

function recur_step(a::Vector, b::Vector; expw)
    expkv_prev, expk_prev = a
    expkv, expk = b 
    @. [expw*expkv_prev + expkv, expw*expk_prev + expk]
end

struct TokenMixing{T}
    Tₖ::AbstractArray{T, 1}
    Tᵥ::AbstractArray{T, 1}
    Tᵣ::AbstractArray{T, 1}
    r_proj
    k_proj
    v_proj
    out_proj 
    # rescale::AbstractArray{T, 1}  # <-- (e^w e^u - 1)
    time_first::AbstractArray{T, 1}
    time_decay::AbstractArray{T, 1}  # <-- w
end

@Flux.functor TokenMixing

TokenMixing(n_embed::Integer) = TokenMixing(
    zeros(Float32, n_embed), # Tₖ
    zeros(Float32, n_embed), # Tᵥ
    zeros(Float32, n_embed), # Tᵣ
    Dense(n_embed, n_embed, bias=false), # r_proj
    Dense(n_embed, n_embed, bias=false), # k_proj
    Dense(n_embed, n_embed, bias=false), # v_proj
    Dense(n_embed, n_embed, bias=false), # out_proj
    # zeros(Float32, n_embed), # rescale
    zeros(Float32, n_embed), # time first
    ones(Float32, n_embed),  # time_decay
)

function (m::TokenMixing)(x::AbstractArray{T,1}, state::State; i) where T
    n_embed = size(x)
    x_prev = @views(state.x_tm[:, i])

    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵥ = time_mix(x, x_prev, m.Tᵥ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ)
    v = m.v_proj(xᵥ)

    expk = @. exp(k)
    expkv = @. expk * v

    a_prev, b_prev = @views(state.a[:, i]), @views(state.b[:, i])
    a, b = recur_step([a_prev, b_prev], [expkv, expk]; expw=exp.(m.time_decay))

    # update state
    @views state.a[:, i] .= a
    @views state.b[:, i] .= b
    @views state.x_tm[:, i] .= x

    rwkv = @. r * (a_prev + exp(m.time_first)*expkv) / (b_prev + exp(m.time_first)*expk)

    m.out_proj(rwkv), state
end

function (m::TokenMixing)(x::AbstractArray{T,2}, state::State; i) where T
    n_embed, n_seq = size(x)

    x_prev = hcat(@views(state.x_tm[:, i]), @views(x[:, 1:end-1]))
    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵥ = time_mix(x, x_prev, m.Tᵥ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ)
    v = m.v_proj(xᵥ)

    expk = @. exp(k)
    expkv = @. expk * v

    step_f = (a, b) -> recur_step(a, b; expw=exp.(m.time_decay))
    substrate = [[@views(expkv[:,i]), @views(expk[:,i])] for i = 1:n_seq]
    ab = accumulate(step_f, substrate)  # |> batch

    # update state
    @views state.x_tm[:, i] .= x[:, end]
    @views state.a[:, i] .= ab[end][1]
    @views state.b[:, i] .= ab[end][2]

    a = [zeros_like(ab[1][1]);; [ab[i][1] for i = 1:n_seq-1]...]
    b = [zeros_like(ab[2][1]);; [ab[i][2] for i = 1:n_seq-1]...]
    rwkv = map(1:n_seq) do i
        @views @. r[:, i] * (a[:, i] + exp(m.time_first)*expkv[:, i]) / (b[:, i] + exp(m.time_first)*expk[:, i])
    end |> batch

    m.out_proj(rwkv), state
end
    
struct ChannelMixing{T}
    Tₖ::AbstractArray{T, 1}  # will be taken out in the future
    Tᵣ::AbstractArray{T, 1}  # will be taken out in the future
    r_proj
    k_proj
    v_proj
end

@Flux.functor ChannelMixing

ChannelMixing(n_embed::Integer) = ChannelMixing(
    zeros(Float32, n_embed), # Tₖ
    zeros(Float32, n_embed), # Tᵣ
    Dense(n_embed, n_embed, bias=false), # r_proj
    Dense(n_embed, n_embed, bias=false), # k_proj
    Dense(n_embed, n_embed, bias=false), # v_proj
)

function (m::ChannelMixing)(x::AbstractArray{T, 1}, state::State; i) where T
    x_prev = @views(state.x_cm[:, i])
    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ) .|> square_relu

    # update state
    @views state.x_cm[:, i] .= x[:, end]

    r .* (m.v_proj(k)), state
end

function (m::ChannelMixing)(x::AbstractArray{T, 2}, state::State; i) where T
    n_embed, n_seq = size(x)

    x_prev = @views(state.x_cm[:, i])
    if size(x, 2) > 1
        x_prev = hcat(x_prev, @views(x[:, 1:end-1]))
    end
    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ) .|> square_relu

    # update state
    @views state.x_cm[:, i] .= x[:, end]

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
    TokenMixing(n_embed),
    LN(n_embed),
    ChannelMixing(n_embed),
)

function (m::Block)(x, state; i)
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

(m::RWKV)(x::Union{Integer, AbstractArray{T, 1}}, state::State) where T = begin
    x = m.embedding(x)
    x = m.ln_init(x)
    for i in 1:length(m.blocks)
        x, state = m.blocks[i](x, state; i=i)
    end
    x = m.ln_final(x)

    # x: [n_embed, n_seq]
    x = m.lm_head.weight' * x

    x, state
end