include("lib.jl")
include("data.jl")

using Flux.Losses
using OneHotArrays
using Wandb, Dates, Logging
using CUDA

lg = WandbLogger(
    project = "nano-llm.jl",
    name = "gpt2-$(now())",
    config = Dict(
        "learning_rate" => 3e-4,
        "batchsize" => 4,
        "epochs" => 100,
        "dataset" => "data.jsonl",
        "use_cuda" => true,
        "ctx_len" => 256,
    ),
)
global_logger(lg)

data_file = get_config(lg, "dataset")
ts = TextSplitter(get_config(lg, "ctx_len")+1, 64)

n_vocab = ts.tokenizer.n_vocab
n_embed = 384
n_layer = 6
n_head  = 6
ctx_len = get_config(lg, "ctx_len")

function loss_func(y_pred, y)
    return logitcrossentropy(y_pred, onehotbatch(y, 1:n_vocab))
end

if get_config(lg, "use_cuda") && CUDA.functional()
    CUDA.allowscalar(false)
    @info "Using CUDA"
    device = gpu
else
    @info "Using CPU"
    device = identity
end
# setup optimizer

model = GPT2(n_vocab, n_embed, n_head, n_layer, ctx_len) |> device
opt_state = Flux.setup(Adam(get_config(lg, "learning_rate")), model)

for epoch in 1:get_config(lg, "epochs")
    batches = jsonl_reader(data_file) |> ch->batch_sampler(ch, ts)
    i_batch = 1
    for (x, y) in batches
        x = x |> device
        y = y |> device
        val, grads = Flux.withgradient(model) do m
            y_pred = m(x)
            loss_func(y_pred, y)
        end
        Flux.update!(opt_state, model, grads[1])
        if i_batch % 10 == 0
            @info "metrics" loss=val
        end
        i_batch += 1
    end
end

close(lg)