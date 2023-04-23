include("lib.jl")
include("utils.jl")

function generate(gpt2, inputs; n_tokens_to_generate=10)
    for _ in 1:n_tokens_to_generate # auto-regressive decode loop
        inputs_ = @views inputs[:,:]
        logits = gpt2(inputs_) # model forward pass
        next_id = argmax(logits[:,end,1])
        inputs = [inputs ; next_id]
    end
    return inputs[length(inputs)-n_tokens_to_generate+1:end] # only return generated ids
end

models_dir = "gpt2_124m"

# load encoder, hparams, and params from the released open-ai gpt-2 files
encoder = get_encoder()

# encode the input string using the BPE tokenizer
prompt = "Alan Turing theorized that computers would one day become"
n_tokens_to_generate = 10

# julia indexing starts at 1
input_ids = encoder.encode(prompt) .+ 1


# make sure we are not surpassing the max sequence length of our model
# @assert length(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
gpt2 = gpt2_from_tf_ckpt(models_dir)

# generate output ids
output_ids = generate(gpt2, input_ids; n_tokens_to_generate)

# decode the ids back into a string
# -1 is going back to python indexing
output_text = encoder.decode(output_ids .- 1)

println(prompt)
println(output_text)