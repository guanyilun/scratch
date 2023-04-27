include("utils.jl")

# be aware that it may default to f16
model = opt_from_pytorch_ckpt("attempt4/pytorch_model.bin", n_layers=12, n_head=12) |> gpu
tokenizer = get_galactica_tokenizer()

function generate(model, inputs; n_tokens_to_generate=10)
    for _ in 1:n_tokens_to_generate # auto-regressive decode loop
        inputs_ = @views inputs[:,:]
        logits = model(inputs_) # model forward pass
        next_id = argmax(logits[:,end,1])
        inputs = [inputs ; next_id]
    end
    return inputs[length(inputs)-n_tokens_to_generate+1:end] # only return generated ids
end

n_tokens_to_generate = 20

prompt = "The Transformer architecture [START_REF]"
input_ids = Int32.(tokenizer.encode(prompt) .+ 1)
input_ids = [1 ; input_ids]
input_ids = input_ids |> gpu
output_ids = generate(model, input_ids; n_tokens_to_generate)

@show prompt;
output_text = tokenizer.decode(output_ids .- 1)
