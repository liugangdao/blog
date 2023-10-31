

# 训练一个玩Mario的智能体
> 本文是**强化学习系列**的第一篇博客，是本人强化学习的第一次尝试。

## 环境安装

导入本案例所需要的包

`
!pip install gym-super-mario-bros==7.4.0
`

```
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
```

## 强化学习定义

**环境 (Environment)** 代理与之交互并从中学习的世界。

>动作 (Action) $a$：代理对环境的响应。所有可能的动作的集合被称为动作空间 (action-space)。

>状态 (State) $s$：环境的当前特征。环境可能处于的所有可能状态的集合被称为状态空间 (state-space)。

>奖励 (Reward) $r$：奖励是环境对代理的关键反馈。它驱使代理学习并改变其未来的行动。多个时间步长的奖励聚合称为 回报 (Return)

>最优动作值函数 (Optimal Action-Value function) $Q^*(s,a)$：如果从状态 $s$ 开始，采取任意动作 $a$，然后对于每个未来的时间步都采取最大化回报的动作，给出相应的预期回报。$Q$ 可以说是表示了某个状态下动作的“质量”。我们尝试逼近这个函数。

### 环境初始化

环境