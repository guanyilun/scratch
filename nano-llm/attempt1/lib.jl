using Statistics
using LinearAlgebra
using MLUtils

gelu(x) = Float32(0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))))

function softmax(x; dims=1)
    exp_x = exp.(x .- maximum(x, dims=dims))
    exp_x ./ sum(exp_x, dims=dims)
end

function layer_norm(x; γ, β, ϵ=1e-5, dims=1)
    μ = mean(x, dims=dims)
    σ² = var(x, dims=dims)
    Float32.(γ .* (x .- μ) ./ (σ² .+ ϵ).^0.5 .+ β)
end

function linear(x; W, b)
    W * x .+ b
end

function ffn(x; c_fc, c_proj)
    x = gelu.(linear(x; c_fc...))
    linear(x; c_proj...)
end 

function self_attn(q, k, v, mask) # [n_interim, n_seq] -> [n_seq, n_seq] x [n_seq, n_interim] -> [n_seq, n_interim]
    attention = q' * k ./ Float32(sqrt(size(q, 1)))
    attention .+= mask
    attention .= softmax(attention; dims=2)  # dims=2 is important!
    v * attention'
end

function mha(x; c_attn, c_proj, n_head) # [n_embed, n_seq] -> [n_embed, n_seq]
    n_embed, n_seq = size(x)
    n_interim = n_embed ÷ n_head

    # qkv projection
    qkv = linear(x; c_attn...) # [n_embed, n_seq] -> [n_embed, n_seq]
    q, k, v = chunk(qkv, 3, dims=1) # [n_embed, n_seq] -> [n_embed, n_seq] x 3

    @assert size(q) == size(k) == size(v) == (n_embed, n_seq)

    # split into heads
    q = reshape(q, n_interim, n_head, n_seq) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]
    k = reshape(k, n_interim, n_head, n_seq) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]
    v = reshape(v, n_interim, n_head, n_seq) # [n_embed, n_seq] -> [n_interim, n_head, n_seq]

    # change order of dimensions
    q = permutedims(q,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]
    k = permutedims(k,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]
    v = permutedims(v,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = Float32.((1 .- tril!(fill(1f0, (n_seq, n_seq)))) * -1e10) # [n_seq, n_seq]

    x = zeros(eltype(q), n_interim, n_seq, n_head)
    heads = []
    for i in 1:n_head
        @views push!(heads, self_attn(q[:,:,i], k[:,:,i], v[:,:,i], causal_mask))
    end
    x = cat(heads..., dims=1)

    x = linear(x; c_proj...) # [n_embed, n_seq] -> [n_seq, n_embd]
    x
end

function transformer_block(x; mlp, attn, ln1, ln2, n_head)
    x = x .+ mha(layer_norm(x; ln1...); attn..., n_head=n_head)
    x = x .+ ffn(layer_norm(x; ln2...); mlp...)
    x
end

function gpt2(inputs; wte, wpe, blocks, ln_f, n_head)
    # token + positional embeddings
    x = wte[:,inputs] .+ wpe[:,collect(1:length(inputs))] # [n_seq] -> [n_embed, n_seq]

    for i in eachindex(blocks)
        x = transformer_block(x; blocks[i]..., n_head=n_head)
    end

    x = layer_norm(x; ln_f...) # [n_embed, n_seq]

    # [n_embed, n_vocab]' x [n_embed, n_seq] -> [n_vocab, n_seq]
    wte' * x  
end

function generate(inputs; params, n_tokens_to_generate)
    for _ in 1:n_tokens_to_generate # auto-regressive decode loop
        logits = gpt2(inputs; params...) # model forward pass
        next_id = argmax(logits[:,end]) # greedy sampling
        push!(inputs, next_id) # append prediction to input
    end
    return inputs[(length(inputs)-n_tokens_to_generate)+1:end] # only return generated ids
end
