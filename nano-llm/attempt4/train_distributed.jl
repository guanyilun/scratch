using Distributed
using CUDA

addprocs(length(devices()))

@everywhere begin
    using Optimisers, CUDA
    using OneHotArrays
    using Flux.Losses

    include("lib.jl")
    include("data.jl")
    include("train_utils.jl")

    # mini model for demonstration
    model_cfg = (
        n_vocab = 50257,
        n_embed = 10,
        n_head  = 2,
        n_layer = 2,
        ctx_len = 256,
    )

    train_cfg = (
        lr = 3e-4,
        batch_size = 8,
        epochs = 10,
        data_path = "data.jsonl",
    )

    function loss_func(y_pred, y; n_vocab=model_cfg.n_vocab)
       return logitcrossentropy(y_pred, onehotbatch(y, 1:n_vocab))
    end
end


model = GPT2(model_cfg...)
opt = Flux.setup(Optimisers.Adam(train_cfg.lr), model)

ts = TextSplitter(model_cfg.ctx_len+1, 64)

data_ch = RemoteChannel(()->Channel{Tuple}(10), 1)
w2m_chs = [RemoteChannel(()->Channel{Any}(1), p) for p in workers()]
m2w_chs = [RemoteChannel(()->Channel{Any}(1), 1) for p in workers()]

# setup data loading
function start_dataloader(ch; data_file=train_cfg.data_path, ts=ts)
    @info "Starting dataloader"
    batches = jsonl_reader(data_file) |> c->batch_sampler(c, ts; batch_size=train_cfg.batch_size)
    for (x, y) in batches
        put!(ch, (x, y))
    end
end
errormonitor(@async start_dataloader(data_ch))


function start_syncgrad(w2m_chs, m2w_chs)
    @info "Starting syncgrad"
    while true
        all(map(isready, w2m_chs)) || continue
        grads = map(take!, w2m_chs) .|> cpu

        grads = reduce(grads) do grad1, grad2
            fmap(grad1, grad2) do g1, g2
                isnothing(g1) && return g2
                isnothing(g2) && return g1
                g1 .+ g2
            end
        end

        grads = fmap(grads) do g
            isnothing(g) && return nothing
            g ./ length(w2m_chs)
        end

        foreach(m2w_chs) do ch
            put!(ch, grads)
        end
    end
end
errormonitor(@async start_syncgrad(w2m_chs, m2w_chs))

# setup training loop
asyncmap(workers(), devices(), w2m_chs, m2w_chs) do p, d, w2m_ch, m2w_ch
    remotecall_wait(p, model, opt, data_ch) do model, opt, data_ch
        @info "Worker $p on device $d"
        device!(d)

        # setup model
        model = model |> gpu
        opt = opt |> gpu

        while true
            x, y = take!(data_ch)
            x = x |> gpu
            y = y |> gpu
            grad, = gradient(model) do model
                logits = model(x)
                loss_func(logits, y)
            end

            @info "Worker $p sending grads"
            put!(w2m_ch, grad)

            grad = take!(m2w_ch) |> gpu
            @info "Worker $p received aggregated grads"

            opt, model = Optimisers.update!(opt, model, grad)
            @info "Worker $p updated model"
        end
    end
end
