"""
In this script I explore a pattern to build non-trivial multi-stage
computation, where each stage not only has the option to pass result
to the next stage, but also can signal for reprocessing in the previous
stage.

"""
#%%
from functools import partial
from dataclasses import dataclass
from typing import Any
from functools import reduce

identity = lambda x: x

@dataclass
class RunStage:
    ir: Any
    stage: Any
    succeed: Any
    
@dataclass
class Lower:
    ir: Any

def run_stage(ir, stage, succeed):
    return RunStage(ir, stage, succeed(Lower(ir)))

stages = ['stage1', 'stage2', 'stage3', 'stage4']

def build_multi_stages(stages):
    builder = lambda succeed, stage: partial(run_stage, stage=stage, succeed=succeed)
    f = reduce(builder, stages[::-1], identity)
    return f

#%%
from equinox import tree_pprint
tree_pprint(build_multi_stages(stages)("lvl1"))

"""
Output:

RunStage(
  ir='lvl1',
  stage='stage1',
  succeed=RunStage(
    ir=Lower(ir='lvl1'),
    stage='stage2',
    succeed=RunStage(
      ir=Lower(ir=Lower(ir='lvl1')),
      stage='stage3',
      succeed=RunStage(
        ir=Lower(ir=Lower(ir=Lower(ir='lvl1'))),
        stage='stage4',
        succeed=Lower(ir=Lower(ir=Lower(ir=Lower(ir='lvl1'))))
      )
    )
  )
)
"""

#%%
# actually implement a working example
from itertools import count

@dataclass
class L1IR:
    data: Any
    choice: Any
    
@dataclass
class L2IR:
    data: Any
    choice: Any

@dataclass
class L3IR:
    data: Any
    choice: Any

from enum import Enum, auto
class SignalMode(Enum):
    Next = auto()

class Signal(Exception):
    def __init__(self, mode: SignalMode):
        self.mode = mode

class Stage1:
    def lower(self, expr: L1IR) -> L2IR:
        for c in count():
            print(f"Stage1: Trying choice {c}")
            yield L2IR(expr.data, c)
    def lift(self, expr: L2IR) -> L1IR:
        return L1IR(expr.data, expr.choice)

class Stage2:
    def lower(self, expr: L2IR) -> L3IR:
        data, choice = expr.data, expr.choice
        # report failure when choice is not 3,
        # ask for a different choice
        if choice != 3:
            print(f"Stage2: Choice {choice} failed, signaling up")
            raise Signal(SignalMode.Next)
        yield L3IR(data, choice)
    def lift(self, expr: L3IR) -> L2IR:
        return L2IR(expr.data, expr.choice)
    
def run_stage(ir, stage, succeed):
    for ir in stage.lower(ir):
        try:
            return succeed(ir) 
        except Signal as e:
            if e.mode == SignalMode.Next:
                continue
            else:
                raise
    raise RuntimeError("No solution found in Stage {}".format(stage))

stages = [Stage1(), Stage2()]
res = build_multi_stages(stages)(L1IR("data", 0))

"""
Output:
Stage1: Trying choice 0
Stage2: Choice 0 failed, signaling up
Stage1: Trying choice 1
Stage2: Choice 1 failed, signaling up
Stage1: Trying choice 2
Stage2: Choice 2 failed, signaling up
Stage1: Trying choice 3


"""
