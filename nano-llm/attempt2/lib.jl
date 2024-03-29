using MLUtils
using NNlib
using Flux
using LinearAlgebra
using Statistics

struct LN
    γ
    β
    ϵ
    dims
end
function (m::LN)(x)
    μ = mean(x, dims=m.dims)
    σ² = var(x, dims=m.dims)
    eltype(x).(m.γ .* (x .- μ) ./ (σ² .+ m.ϵ).^0.5 .+ m.β)
end
function LN(γ, β)
    LN(γ, β, 1e-5, 1)
end
function LN(n::Integer)
    LN(Flux.glorot_uniform(n), zeros(n))
end
@Flux.functor LN
Flux.trainable(m::LN) = (m.γ, m.β)


struct FFN
    fc
    proj
end
(m::FFN)(x) = Chain(m.fc, gelu, m.proj)(x)
@Flux.functor FFN
Flux.trainable(m::FFN) = (m.fc, m.proj) # optional

function self_attn(q, k, v, mask) # [n_interim, n_seq] -> [n_seq, n_seq] x [n_seq, n_interim] -> [n_seq, n_interim]
    attention = k' * q ./ eltype(q)(sqrt(size(q, 1)))
    attention .+= mask
    attention .= softmax(attention)  # dims=1 is important, goes with triu mask
    v * attention
end

struct MHA
    attn
    proj
    n_head
end
function (m::MHA)(x::AbstractArray{T,2}) where T
    n_embed, n_seq = size(x)
    n_interim = n_embed ÷ m.n_head

    # qkv projection
    qkv = m.attn(x) # [n_embed, n_seq] -> [n_embed, n_seq]
    q, k, v = chunk(qkv, 3, dims=1) # [n_embed, n_seq] -> [n_embed, n_seq] x 3

    @assert size(q) == size(k) == size(v) == (n_embed, n_seq)

    # split into heads
    q = reshape(q, n_interim, m.n_head, n_seq) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]
    k = reshape(k, n_interim, m.n_head, n_seq) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]
    v = reshape(v, n_interim, m.n_head, n_seq) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]

    # change order of dimensions
    q = permutedims(q,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]
    k = permutedims(k,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]
    v = permutedims(v,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]

    # causal mask to hide future inputs from being attended to
    # both methods work, first one comes from NNlib
    causal_mask = (one(T) .- make_causal_mask(q, dims=2)) .* T(-1e10)
    # causal_mask = (one(T) .- triu(fill!(similar(q, Bool, (n_seq, n_seq)), true))) * T(-1e10) # [n_seq, n_seq]

    heads = []
    for i in 1:m.n_head
        @views push!(heads, self_attn(q[:,:,i], k[:,:,i], v[:,:,i], causal_mask))
    end

    cat(heads..., dims=1) |> m.proj
end
@Flux.functor MHA
Flux.trainable(m::MHA) = (m.attn, m.proj)

struct TransformerDecoderBlock
    mha
    mlp
    ln1
    ln2
end
function (m::TransformerDecoderBlock)(x)
    x .+= m.mha(m.ln1(x))
    x .+= m.mlp(m.ln2(x))
    x
end
@Flux.functor TransformerDecoderBlock
Flux.trainable(m::TransformerDecoderBlock) = (m.mha, m.mlp, m.ln1, m.ln2) # optional

struct GPT2 # GPT2 model
    wte
    wpe
    blocks
    ln_f
end
function (m::GPT2)(inputs)
    # token + positional embeddings
    x = m.wte.weight[:,inputs] .+ m.wpe.weight[:,collect(1:length(inputs))] # [n_seq] -> [n_embed, n_seq]
    x = Chain(m.blocks..., m.ln_f)(x) # [n_embed, n_seq]

    # [n_embed, n_vocab]' x [n_embed, n_seq] -> [n_vocab, n_seq]
    m.wte.weight' * x
end
@Flux.functor GPT2
Flux.trainable(m::GPT2) = (m.wte, m.wpe, m.blocks, m.ln_f)  # optional

struct GPT2Config
    vocab_size::Int
    n_layer::Int
    n_head::Int
    n_embed::Int
    ctx_len::Int
end

function GPT2(config::GPT2Config)
    wte = Dense(config.vocab_size => config.n_embed; bias=false) # [n_embed, n_vocab]
    wpe = Dense(config.ctx_len => config.n_embed; bias=false)    # [n_embed, n_seq]

    blocks = []
    for _ in 1:config.n_layer
        push!(blocks, TransformerDecoderBlock(
            MHA(
                Dense(config.n_embed => 3*config.n_embed), # attn
                Dense(config.n_embed => config.n_embed),   # proj
                config.n_head
            ),
            FFN(
                Dense(config.n_embed => 3*config.ctx_len), # fc
                Dense(3*config.ctx_len => config.n_embed)  # proj
            ),
            LN(config.n_embed),
            LN(config.n_embed)
        ))
    end

    GPT2(wte, wpe, blocks, LayerNorm(config.n_embed))
end
