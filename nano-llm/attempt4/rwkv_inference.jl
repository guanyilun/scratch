include("rwkv.jl")
include("rwkv_utils.jl")

tokenizer = get_tokenizer()
model = rwkv_from_pth("RWKV-4-Pile-430M-20220808-8066.pth"; n_layer=24)

function generate(model, prompt, n_tokens=50; top_p=0.99, temperature=1.0)
    input_ids = tokenizer.encode(prompt).ids .+ 1

    state = nothing
    for input_id in input_ids[1:end-1]
        out, state = model(input_id, state)
    end
    println(prompt)

    input_id = input_ids[end]
    for i = 1:n_tokens
        out, state = model(input_id, state)
        out_id = sample_logits(out; top_p=top_p, temperature=temperature)
        print(tokenizer.decode([out_id-1]))
        input_id = out_id
    end
    println()
end

prompt = "Pumas are large, cat-like animals found in" 
generate(model, prompt, 200)