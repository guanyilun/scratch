using PyPlot

_g(a=0, b=0, c=0; rule=rule) = rule[c*2^2 + b*2 + a+1]
function next_seq(seq; rule)
    res = zeros(Int, length(seq)+1)
    for i = 1:length(seq)
        if i == 1
            res[i+1] = _g(0, 0, seq[i]; rule=rule)
        elseif i == 2
            res[i+1] = _g(0, seq[i-1], seq[i]; rule=rule)
        else
            res[i+1] = _g(seq[i-2], seq[i-1], seq[i]; rule=rule)
        end
    end
    res
end

function gen_seqs(nsteps; init=[0,1,0,0,1,1], rule)
    seqs = [init]
    x = init
    for i = 1:nsteps
        x = next_seq(x; rule=rule)
        push!(seqs, x)
    end
    seqs
end

function show_rule(rule)
    for a = 0:1
        for b = 0:1
            for c = 0:1
                println("g($a, $b, $c) = ", _g(a, b, c; rule=rule))
            end
        end
    end
end

rule = [1 0 0 0 1 1 0 0]
show_rule(rule)
gen_seqs(10; rule=rule)