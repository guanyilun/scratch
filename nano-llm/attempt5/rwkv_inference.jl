include("rwkv.jl")
include("rwkv_utils.jl")

tokenizer = get_tokenizer()
model = rwkv_from_pth("RWKV-4-Pile-430M-20220808-8066.pth"; n_layer=24)
model = model |> gpu

function generate(model, prompt, n_tokens=50; top_p=0.99, temperature=1.0, use_argmax=false)
    input_ids = tokenizer.encode(prompt).ids .+ 1

    state = State(size(model.embedding.weight, 1), length(model.blocks)) |> gpu

    out, state = model(input_ids[1:end-1], state)
    # for input_id in input_ids[1:end-1]
    #     out, state = model(input_id, state)
    # end

    println("-------------------------")
    println(prompt)
    println("-------------------------")

    input_id = input_ids[end]
    for i = 1:n_tokens
        out, state = model(input_id, state)
        out_id = sample_logits(out; top_p=top_p, temperature=temperature, use_argmax=use_argmax)
        print(tokenizer.decode([out_id-1]))
        input_id = out_id
    end
    println()
    state
end

prompt = "The hunt for the puma began in a small village where a woman picking blackberries saw 'a large cat' only five yards away from her. It";
generate(model, prompt, 100; use_argmax=false, top_p=0.999);
