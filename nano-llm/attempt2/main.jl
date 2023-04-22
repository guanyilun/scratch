include("lib.jl")
include("utils.jl")

models_dir = "gpt2_124m"

# load encoder, hparams, and params from the released open-ai gpt-2 files
# hparams, params = hparams_and_params_from_tf_checkpoint(models_dir)
encoder = get_encoder()

# display(inspect_params(params))

# encode the input string using the BPE tokenizer
prompt = "Alan Turing theorized that computers would one day become"
n_tokens_to_generate = 10

# julia indexing starts at 1
input_ids = encoder.encode(prompt) .+ 1

# make sure we are not surpassing the max sequence length of our model
# @assert length(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
# gpt2 = GPT2(GPT2Config(50257, 24, 16, 768, 1024))
gpt2 = gpt2_from_tf_ckpt(models_dir)

# generate output ids
output_ids = generate(gpt2, input_ids; n_tokens_to_generate)

# decode the ids back into a string
# -1 is going back to python indexing
output_text = encoder.decode(output_ids .- 1)

println(prompt)
println(output_text)