# transformer
Learn and experimenting with transformer by reimplementing
a minimum version of gpt-2 in julia following the implementation
in [picoGPT](https://github.com/jaymody/picoGPT/).

## Attempt 1
In my first attempt, I will implement most of the
building blocks from scratch following picoGPT. I avoided using
existing library such as Flux or NNlib. 
- Learned how to work with column major convention in julia
- Understood the basic building blocks
- Still need to learn how to train such models
- TODO: Outputs still look wrong, need to investigate

## Attempt 2 (in progress)
Ideas for the second attempt
- use more building blocks from Flux and NNlib