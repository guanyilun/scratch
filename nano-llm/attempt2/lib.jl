using Statistics
using LinearAlgebra
using MLUtils
# using TensorOperations
# using Tullio

function layer_norm(x; γ, β, ϵ=1e-5, dims=1)
    μ = mean(x, dims=dims)
    σ² = var(x, dims=dims)
    (x .- μ) ./ (σ² .+ ϵ).^0.5 .* γ .+ β
end

function linear(x; W, b)
    W * x .+ b
end

function ffn(x; c_fc, c_proj)
    x = gelu.(linear(x; c_fc...))
    linear(x; c_proj...)
end 

function self_attn(q, k, v, mask) # [n_interim, n_seq] -> [n_seq, n_seq] x [n_seq, n_interim] -> [n_seq, n_interim]
    return v * softmax(q' * k ./ sqrt(size(q, 1)) + mask)
end

function mha(x; c_attn, c_proj, n_head) # [n_embed, n_seq] -> [n_embed, n_seq]
    n_embed, n_seq = size(x)
    n_interim = n_embed ÷ n_head

    # qkv projection
    qkv = linear(x; c_attn...) # [n_embed, n_seq] -> [n_embed, n_seq]
    q, k, v = chunk(qkv, 3, dims=1) # [n_embed, n_seq] -> [n_embed, n_seq] x 3

    # split into heads
    q = reshape(q, n_interim, n_head, n_seq) # [n_embed, n_seq] -> [n_seq, n_interim, n_head]
    k = reshape(k, n_interim, n_head, n_seq) # [n_embed, n_seq] -> [n_seq, n_interim, n_head]
    v = reshape(v, n_interim, n_head, n_seq) # [n_embed, n_seq] -> [n_seq, n_interim, n_head]

    # change order of dimensions
    q = permutedims(q,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]
    k = permutedims(k,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]
    v = permutedims(v,(1,3,2)) # [n_interim, n_head, n_seq] -> [n_interim, n_seq, n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 .- tril!(zeros(n_seq, n_seq))) * -1e10 # [n_seq, n_seq]

    x = zeros(n_interim, n_seq, n_head)
    # this should be equivalent to the commented out line below
    # @tullio x[a,b,c] = self_attn(q[:,:,c], k[:,:,c], v[:,:,c], causal_mask)[a,b]
    for i in 1:n_head
        x[:,:,i] = @views self_attn(q[:,:,i], k[:,:,i], v[:,:,i], causal_mask)
    end
    x = permutedims(x, (1,3,2))
    x = reshape(x, n_embed, n_seq) # [n_seq, n_interim, n_head] -> [n_seq, n_embed]
    x = linear(x; c_proj...) # [n_seq, n_embd] -> [n_seq, n_embd]
    x
end

function transformer_block(x; mlp, attn, ln1, ln2, n_head)
    x = x + mha(layer_norm(x; ln1...); attn..., n_head=n_head)
    x = x + ffn(layer_norm(x; ln2...); mlp...)
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