using Test
include("lib.jl")

# Test the gelu function
@testset "gelu" begin
    x = 0.0
    @test gelu(x) ≈ 0.0

    x = 1.0
    @test gelu(x) ≈ 0.841192

    x = -1.0
    @test gelu(x) ≈ -0.15880800939172324
end

# Test the softmax function
@testset "softmax" begin
    x = [1.0, 2.0, 3.0]
    @test softmax(x) ≈ [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
end

# Test the layer_norm function
@testset "layer_norm" begin
    x = [1.0 2.0; 3.0 4.0]
    γ = [1.0, 1.0]
    β = [0.0, 0.0]
    @test layer_norm(x; γ=γ, β=β) ≈ [-0.7071050134262237 -0.7071050134262237; 0.7071050134262237 0.7071050134262237]
end

# Test the linear function
@testset "linear" begin
    x = [1.0 2.0; 3.0 4.0]
    W = [1.0 2.0; 3.0 4.0]
    b = [0.0, 1.0]
    @test linear(x; W=W, b=b) ≈ [7.0 10.0; 16.0 23.0]
end

# You can create similar test cases for other functions as well.
# Note that testing functions like 'mha', 'transformer_block', and 'gpt2' might require you to create dummy parameters and inputs.

# For example, testing the mha function:
@testset "mha" begin
    x = rand(8, 4)
    c_attn = Dict(:W=>rand(24, 8), :b=>rand(24))
    c_proj = Dict(:W=>rand(8, 8), :b=>rand(8))
    n_head = 2
    output = mha(x; c_attn=c_attn, c_proj=c_proj, n_head=n_head)
    @test size(output) == size(x)
end

@testset "transformer_block" begin
    x = rand(8, 4)
    mlp = Dict(
        :c_fc => Dict(:W=>rand(8, 8), :b=>rand(8)), 
        :c_proj => Dict(:W=>rand(8, 8), :b=>rand(8)))
    attn = Dict(
        :c_attn => Dict(:W=>rand(24, 8), :b=>rand(24)),
        :c_proj => Dict(:W=>rand(8, 8), :b=>rand(8))
    )
    ln1 = Dict(:γ=>rand(8), :β=>rand(8))
    ln2 = Dict(:γ=>rand(8), :β=>rand(8))
    output = transformer_block(x; mlp=mlp, attn=attn, ln1=ln1, ln2=ln2, n_head=2)
    @test size(output) == size(x)
end

@testset "gpt2 and gen" begin
    n_vocab = 10
    n_embed = 8
    n_seq = 4
    n_head = 2
    n_block = 1
    n_context = 20
    wte = rand(n_embed, n_vocab)
    wpe = rand(n_embed, n_context)

    input = collect(1:n_seq)

    blocks = [
        Dict(
            :mlp => Dict(
                :c_fc => Dict(:W=>rand(8, 8), :b=>rand(8)), 
                :c_proj => Dict(:W=>rand(8, 8), :b=>rand(8))), 
            :attn => Dict(
                :c_attn => Dict(:W=>rand(24, 8), :b=>rand(24)),
                :c_proj => Dict(:W=>rand(8, 8), :b=>rand(8))
            ), 
            :ln1 => Dict(:γ=>rand(8), :β=>rand(8)), 
            :ln2 => Dict(:γ=>rand(8), :β=>rand(8))
        )
    ]
    ln_f = Dict(:γ=>rand(8), :β=>rand(8))
    output = gpt2(input; wte=wte, wpe=wpe, blocks=blocks, ln_f=ln_f, n_head=n_head)

    # (n_vocab, n_seq) every token predicts the next token in the form of a pdf over the vocabulary
    @test size(output) == (10, 4)

    output_ids = generate(input; params=Dict(:wte=>wte, :wpe=>wpe, :blocks=>blocks, :ln_f=>ln_f, :n_head=>n_head), n_tokens_to_generate=5)
    @test length(output_ids) == 5
end