include("rwkv.jl")
include("rwkv_utils.jl")

model = rwkv_from_pth("RWKV-4-Pile-430M-20220808-8066.pth"; n_layer=24)

##
using PyPlot

##
fig = figure()
for i = 1:24
    plot(exp.(model.blocks[i].token_mixing.time_decay), label="Layer $i", alpha=0.3)
end
# yscale("log")
ylabel("time decay")
display(fig)

##

fig = figure()
for i = 1:24
    plot(model.blocks[i].token_mixing.time_first, label="Layer $i", alpha=0.3)
end
# yscale("log")
ylabel("time first")
display(fig)

##
fig = figure()
for i = 1:24
    ax = fig.add_subplot(4, 6, i)
    data = reshape(model.blocks[i].token_mixing.time_first, 32, 32)
    ax.imshow(data)
    ax.axis("off")
    ax.set_title("Layer $i")
end
suptitle("Time decay")
tight_layout()
display(fig)


##
fig = figure()
for i = 1:24
    ax = fig.add_subplot(4, 6, i)
    data = reshape(exp.(model.blocks[i].token_mixing.time_decay), 32, 32)
    ax.imshow(data)
    ax.axis("off")
    ax.set_title("Layer $i")
end
suptitle("Time decay")
tight_layout()
display(fig)

##
fig = figure()
for i = 1:24
    ax = fig.add_subplot(4, 6, i)
    data = reshape(exp.(model.blocks[i].token_mixing.Tᵥ), 32, 32)
    ax.imshow(data)
    ax.axis("off")
    ax.set_title("Layer $i")
end
suptitle("Time mixing v")
tight_layout()
display(fig)

##
fig = figure()
for i = 1:24
    ax = fig.add_subplot(4, 6, i)
    data = reshape(exp.(model.blocks[i].token_mixing.Tᵣ), 32, 32)
    ax.imshow(data)
    ax.axis("off")
    ax.set_title("Layer $i")
end
suptitle("Time mixing r")
tight_layout()
display(fig)

## 

function visualize_prompt(prompt)
    tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(prompt).ids .+ 1
    embeds = model.embedding(input_ids)
    n_token = length(input_ids)
    ncol = 6
    nrow = Int(ceil(n_token/ncol))
    fig, axes = subplots(nrow, ncol, squeeze=true)
    @show typeof(axes)
    for (i, ax) in enumerate(permutedims(axes, (2,1)))
        ax.axis("off")
        if i > n_token
            continue
        end
        token = input_ids[i]
        ax.imshow(reshape(embeds[:,i], (32,32)), norm="linear")
        ax.set_title(tokenizer.decode([token-1]), fontsize=8)
    end
    display(fig)
end

##
prompt = "Physics is the study of the fundamental constituents of matter and energy and the interactions between them."
visualize_prompt(prompt)