using Flux
using Statistics
using ChainRulesCore
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

struct MHA
    attn
    proj
    n_head
end
@Flux.functor MHA
Flux.trainable(m::MHA) = (attn=m.attn, proj=m.proj)
MHA(n_embed::Integer, n_head::Integer) = MHA(Dense(n_embed, 3*n_embed), Dense(n_embed, n_embed), n_head)
function (m::MHA)(x::AbstractArray{T,3}) where T
    n_embed, n_seq, n_batch = size(x)
    n_interim = n_embed ÷ m.n_head

    # qkv projection
    qkv = m.attn(x) # [n_embed, n_seq, n_batch] -> [3*n_embed, n_seq, n_batch]
    q, k, v = chunk(qkv, 3, dims=1) # [n_embed, n_seq, n_batch] -> [n_embed, n_seq, n_batch] x 3

    @assert size(q) == size(k) == size(v) == (n_embed, n_seq, n_batch)

    # split into heads
    q = reshape(q, n_interim, m.n_head, n_seq, n_batch) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]
    k = reshape(k, n_interim, m.n_head, n_seq, n_batch) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]
    v = reshape(v, n_interim, m.n_head, n_seq, n_batch) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]

    # self attention
    q = permutedims(q, (1, 3, 2, 4)) # [n_interim, n_head, n_seq, n_batch] -> [n_interim, n_seq, ...]
    k = permutedims(k, (3, 1, 2, 4)) # [n_interim, n_head, n_seq, n_batch] -> [n_seq, n_interim, ...]
    v = permutedims(v, (1, 3, 2, 4)) # [n_interim, n_head, n_seq, n_batch] -> [n_interim, n_seq, ...]

    mask = make_mask(q, dims=2)
    x = self_attn(q, k, v, mask)
    x = permutedims(x, (1, 3, 2, 4)) # [n_interim, n_seq, n_head, n_batch] -> [n_interim, n_head, n_seq, n_batch]
    x = reshape(x, n_embed, n_seq, n_batch) # [n_interim, n_head, n_seq, n_batch] -> [n_embed, n_seq, n_batch]

    m.proj(x)
end

function self_attn(q, k, v, mask) # [n_interim, n_seq] -> [n_seq, n_seq] x [n_seq, n_interim] -> [n_seq, n_interim]
    score = softmax(batched_mul(k, q) ./ eltype(q)(sqrt(size(q, 1))) .+ mask)
    batched_mul(v, score)
end

struct FFN
    fc
    proj
end
FFN(n_embed::Integer) = FFN(Dense(n_embed, 3*n_embed), Dense(3*n_embed, n_embed))
(m::FFN)(x) = Chain(m.fc, gelu, m.proj)(x)
@Flux.functor FFN
Flux.trainable(m::FFN) = (fc=m.fc, proj=m.proj)

struct TransformerDecoderBlock
    mha
    mlp
    ln1
    ln2
end
TransformerDecoderBlock(n_embed::Integer, n_head::Integer) = TransformerDecoderBlock(MHA(n_embed, n_head), FFN(n_embed), LN(n_embed), LN(n_embed))
function (m::TransformerDecoderBlock)(x)
    x = x + Chain(m.ln1, m.mha)(x)
    x = x + Chain(m.ln2, m.mlp)(x)
    x
end
@Flux.functor TransformerDecoderBlock
Flux.trainable(m::TransformerDecoderBlock) = (mha=m.mha, mlp=m.mlp, ln1=m.ln1, ln2=m.ln2)

struct GPT2
    wte
    wpe
    blocks
    ln_f
end
GPT2(n_vocab::Integer, n_embed::Integer, n_head::Integer, n_layer::Integer, ctx_len::Integer) = GPT2(Embedding(n_vocab, n_embed), Embedding(ctx_len, n_embed), [TransformerDecoderBlock(n_embed, n_head) for _ in 1:n_layer], LN(n_embed))
@Flux.functor GPT2
function (m::GPT2)(inputs::AbstractArray{T,2}) where T
    # token + positional embeddings
    x = m.wte(inputs) .+ m.wpe(collect(1:size(inputs,1))) # [n_seq] -> [n_embed, n_seq]
    x = Chain(m.blocks..., m.ln_f)(x) # [n_embed, n_seq]
    # [n_embed, n_vocab]' x [n_embed, n_seq] -> [n_vocab, n_seq]
    batched_mul(m.wte.weight', x)
end
Flux.trainable(m::GPT2) = (wte=m.wte, wpe=m.wpe, blocks=m.blocks, ln_f=m.ln_f)

# testing
# gpt2 = GPT2(200, 128, 8, 2, 100)
# inputs = reshape(collect(1:12), (3, 4))
# x = gpt2.wte.weight[:, inputs] .+ gpt2.wpe.weight[:, collect(1:size(inputs, 1))]
# x_new = Chain(gpt2.blocks..., gpt2.ln_f)(x)
# batched_mul(gpt2.wte.weight', x_new)
# gpt2(inputs)

make_mask(x::AbstractArray; dims::Integer) = (one(eltype(x)) .- make_causal_mask(x, dims=dims)) .* eltype(x)(-1e10) 
@non_differentiable make_mask(::Any...)
