using CUDA
using Setfield
using BSON: @load, @save
using Flux.Losses
using OneHotArrays
using Wandb, Dates, Logging
using Optimisers

include("lib.jl")
include("data.jl")

@load "backups/gpt2-2023-04-24T22:56:49.349.bson" model
ref_model = model

# keep the order
gpt2_config = (
    n_vocab = 50257,
    n_embd = 768,
    n_head = 12,
    n_layer = 24,
    n_ctx = 256,
)

model = GPT2(gpt2_config...)

# map layer from old model to new model
function layer_mapping(i_new, n_layer_ref, n_layer_new)
    # copy the first 3 layers
    if i_new <= 3
        return i_new
    # copy the last 3 layers
    elseif i_new >= n_layer_new - 3
        return n_layer_ref - (n_layer_new - i_new)
    else
        return mod1(i_new-3, (n_layer_ref - 6)) + 3
    end
end

for i = 1:length(model.blocks)
    global model, ref_model
    i_ref = layer_mapping(i, length(ref_model.blocks), length(model.blocks))
    @info "Copying layer $i_ref from old model to layer $i in new model"
    @set! model.blocks[i] = ref_model.blocks[i_ref]
end

# also use the same embedding
@set! model.embedding = ref_model.embedding

# start training
lg = WandbLogger(
    project = "nano-llm.jl",
    name = "gpt2-tune-$(now())",
    config = Dict(
        "learning_rate" => 3e-4,
        "epochs" => 5,
        "dataset" => "astroph_combined.jsonl",
        "use_cuda" => true,
        "ctx_len" => 256,
        "n_layer" => gpt2_config.n_layer
    ),
)

global_logger(lg)

data_file = get_config(lg, "dataset")
ts = TextSplitter(get_config(lg, "ctx_len")+1, 64)

function loss_func(y_pred, y)
    return logitcrossentropy(y_pred, onehotbatch(y, 1:gpt2_config.n_vocab))
end

if get_config(lg, "use_cuda") && CUDA.functional()
    CUDA.allowscalar(false)
    @info "Using CUDA"
    device = gpu
else
    @info "Using CPU"
    device = identity
end

training_config = (
    batch_size = 8,
    log_per_nbatch = 10,
    save_per_nbatch = 10000,
)

opt_state = Flux.setup(Optimisers.Adam(get_config(lg, "learning_rate")), model)

model = model |> device
opt_state = opt_state |> device

i_batch = 1
for epoch in 1:get_config(lg, "epochs")
    batches = jsonl_reader(data_file) |> ch->batch_sampler(ch, ts; batch_size=training_config.batch_size)
    for (x, y) in batches
        global model, opt_state, i_batch
        x = x |> device
        y = y |> device
        val, grads = Flux.withgradient(model) do m
            y_pred = m(x)
            loss_func(y_pred, y)
        end
        opt_state, model = Optimisers.update!(opt_state, model, grads[1])
        if i_batch % training_config.log_per_nbatch == 0
            @info "metrics" loss=val
            println("Ep $(epoch) Batch $(i_batch) Loss $(val)")
        end
        if i_batch % training_config.save_per_nbatch == 0
            @info "saving model"
            save_model(model, "gpt2-tune-$(now()).bson")
        end
        i_batch += 1
    end
end

close(lg)
