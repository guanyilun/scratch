# transformer
Learn and experimenting with transformer by reimplementing
a minimum version of gpt-2 in julia following the implementation
in [picoGPT](https://github.com/jaymody/picoGPT/).

## Attempt 1
In my first attempt, I will implement most of the
building blocks from scratch following picoGPT. I avoided using
existing library such as Flux or NNlib. 
- [x] Learned how to work with column major convention in julia
- [x] Understood the basic building blocks
- [x] Generate outputs that are consistent with picoGPT
- [x] Realise some trickness in array row / column conversion from python to julia

## Attempt 2 
Ideas for the second attempt
- [x] use more building blocks from Flux and NNlib
- [x] able to load pretrained weights
- [x] able to work with new weights
- [x] generate correct output
- [x] running model on gpu
- [ ] training test

## Attempt 3
- [ ] batched input
- [ ] robust training on given corpus
