using JSON
using PyCall
@pyimport tensorflow as tf

# hacky way to get encoder from python
function get_encoder()
    py"""
    import sys
    sys.path.insert(0, "/home/yilun/work/sketchbook/nano-llm/")
    import encoder
    enc = encoder.get_encoder('gpt2_124m') # not nice
    """
    return py"enc"
end

function hparams_and_params_from_tf_checkpoint(tf_checkpoint_path)
    # load hparams from the json file
    hparams = JSON.parsefile(tf_checkpoint_path*"/hparams.json")
    blocks = []
    for i in 0:hparams["n_layer"]-1
        block = Dict(
            :attn => Dict(
                :c_attn => Dict(
                    :W => permutedims(dropdims(tf.train.load_variable(tf_checkpoint_path, "model/h$i/attn/c_attn/w"), dims=1), (2, 1)),
                    :b => tf.train.load_variable(tf_checkpoint_path, "model/h$i/attn/c_attn/b")
                ),
                :c_proj => Dict(
                    :W => permutedims(dropdims(tf.train.load_variable(tf_checkpoint_path, "model/h$i/attn/c_proj/w"), dims=1), (2, 1)),
                    :b => tf.train.load_variable(tf_checkpoint_path, "model/h$i/attn/c_proj/b")
                )
            ), 
            :ln1 => Dict(
                :γ => tf.train.load_variable(tf_checkpoint_path, "model/h$i/ln_1/g"),
                :β => tf.train.load_variable(tf_checkpoint_path, "model/h$i/ln_1/b")
            ),
            :ln2 => Dict(
                :γ => tf.train.load_variable(tf_checkpoint_path, "model/h$i/ln_2/g"),
                :β => tf.train.load_variable(tf_checkpoint_path, "model/h$i/ln_2/b")
            ),
            :mlp => Dict(
                :c_fc => Dict(
                    :W => permutedims(dropdims(tf.train.load_variable(tf_checkpoint_path, "model/h$i/mlp/c_fc/w"), dims=1), (2, 1)),
                    :b => tf.train.load_variable(tf_checkpoint_path, "model/h$i/mlp/c_fc/b")
                ), 
                :c_proj => Dict(
                    :W => permutedims(dropdims(tf.train.load_variable(tf_checkpoint_path, "model/h$i/mlp/c_proj/w"), dims=1), (2, 1)),
                    :b => tf.train.load_variable(tf_checkpoint_path, "model/h$i/mlp/c_proj/b")
                )
            ) 
        ) 
        push!(blocks, block)
    end

    params = Dict(
        :wte => permutedims(tf.train.load_variable(tf_checkpoint_path, "model/wte"), (2, 1)),
        :wpe => permutedims(tf.train.load_variable(tf_checkpoint_path, "model/wpe"), (2, 1)),
        :blocks => blocks,
        :ln_f => Dict(
            :γ => tf.train.load_variable(tf_checkpoint_path, "model/ln_f/g"),
            :β => tf.train.load_variable(tf_checkpoint_path, "model/ln_f/b")
        ),
        :n_head => hparams["n_head"],
    )
    hparams, params
end


function inspect_params(params; lvl=0, spacing="  ")
    for (k, v) in params
        if isa(v, Dict)
            println(repeat(spacing, lvl), k, ":")
            inspect_params(v; lvl=lvl+1)
        elseif k == :blocks  # hacky
            i = 0
            for v2 in v
                println(repeat(spacing, lvl), k, "[$i]:")
                inspect_params(v2; lvl=lvl+1)
                i += 1
            end
        elseif isa(v, Array)
            println(repeat(spacing, lvl), k, ": ", size(v))
        end
    end
end