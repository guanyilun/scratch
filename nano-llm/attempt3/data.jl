using JSON
using PyCall
using Random

function jsonl_reader(file::AbstractString; text_key::AbstractString="text")
    ch = Channel{String}(0)
    task = @async begin
        open(file) do f
            foreach(x->put!(ch, JSON.parse(x)[text_key]), eachline(f))
        end
    end
    bind(ch, task)
    ch
end

struct TextSplitter
    chunk_size
    chunk_overlap
    tokenizer
end
TextSplitter(chunk_size::Int, chunk_overlap::Int, encoding_name::String) = TextSplitter(chunk_size, chunk_overlap, pyimport("tiktoken").get_encoding(encoding_name))
TextSplitter(chunk_size::Int, chunk_overlap::Int) = TextSplitter(chunk_size, chunk_overlap, "gpt2")
TextSplitter() = TextSplitter(256, 64, "gpt2")
function split_text(splitter::TextSplitter, text::AbstractString)
    input_ids = Int32.(splitter.tokenizer.encode(text))
    start_idx = 1
    stop_idx = min(start_idx + splitter.chunk_size - 1, length(input_ids))
    chunk_ids = input_ids[start_idx:stop_idx]
    splits = []
    while start_idx <= length(input_ids)
        push!(splits, chunk_ids)
        start_idx += splitter.chunk_size - splitter.chunk_overlap
        stop_idx = min(start_idx + splitter.chunk_size - 1, length(input_ids))
        chunk_ids = input_ids[start_idx:stop_idx]
    end
    splits
end

function batch_sampler(ch::Channel, splitter::TextSplitter; batch_size::Int=8, buffer_size::Int=16, shuffle::Bool=true, min_block_size::Int=32)
    out_ch = Channel{Tuple{AbstractArray{Int32,2}, AbstractArray{Int32,2}}}(buffer_size)
    task = @async begin
        batch = []
        for text in ch
            splits = split_text(splitter, text)
            shuffle && shuffle!(splits)
            while length(batch) < batch_size
                if isempty(splits)
                    break
                end
                push!(batch, popfirst!(splits))
                if length(batch) == batch_size
                    min_size = minimum(length.(batch))
                    if min_size < min_block_size
                        # discard bad batch
                        empty!(batch)
                        break
                    else
                        x = hcat(map(x->x[1:min_size-1], batch)...) .+1
                        y = hcat(map(x->x[2:min_size], batch)...) .+1
                        put!(out_ch, (x, y))
                        empty!(batch)
                    end
                end
            end
        end
    end
    bind(out_ch, task)
    out_ch
end

# debugging
# data = jsonl_reader("test.jsonl")
# ts = TextSplitter()
# bs = batch_sampler(data, ts)
# x, y = take!(bs)