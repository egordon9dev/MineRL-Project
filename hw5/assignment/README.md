---
layout:     page
title:      Homework 5, Programming Question
permalink:  /nvxGCX9GqdBkRqanv6bVQS/hw5-coding/
---
# CS 4803/7643 Deep Learning - Coding Questions for HW5

In this homework, we will implement algorithms for two kinds - (1) dynamic programming and (2) reinforcement learning with Q-Learning for solving Markov Decision Processes (MDPs).

Note that this homework is adapted from the [Stanford CS234: Reinforcement Learning Winter 2019 course](http://web.stanford.edu/class/cs234/index.html).

Download the starter code [here]({{site.baseurl}}/assets/f20cs7643_hw5_starter.zip).

## Setup

Python 3.7.X is required for this assignment. Either install it directly or create a virtual environment with conda:

```
conda create -n hw5 python=3.7
```

First, install dependencies in `requirements.txt`

```
pip install -r requirements.txt
```

Then, install PyTorch 1.2 from [pytorch.org](https://pytorch.org) - either the CPU or GPU version depending on what your machine supports.

## Part 1: Dynamic Programming (8 regular points + 4 extra credit points for both CS4803 and CS7643)

Open the jupyter notebook `dynamic_programming/dp.ipynb` and follow the instructions to implement policy iteration (policy evaluation + policy improvement) and value iteration.

## Part 2: Q-Learning and Deep Q-Networks (12 regular points + 2 extra credit points for both CS4803 and CS7643)

Open the jupyter notebook `q_learning/q_learning.ipynb` and follow the instructions to implement parts of the Q-Learning training procedure and two types of functions for Q networks - a linear Q network and a convolutional Q network.

## Submission

First, combine all of your PDFs into one PDF, in the following order:

1. Your solutions to questions in PS5
2. Your `dp.ipynb` notebook converted into a PDF(in `dynamic_programming/dp.ipynb`, make sure you run the entire notebook with the `RENDER_ENV` variable set to `False`)
3. Your `q_learning.ipynb` notebook converted into a PDF

This PDF will be submitted under the `HW5` assignment in Gradescope.

Run `collect_submission.sh`

```
./collect_submission.sh
```

which should package your implementations in a ZIP file, as well as your PDFs of the notebooks.
Submit this ZIP to the `HW5 Code` designation in Gradescope.



#### References:

1. [CS234: Reinforcement Learning Winter 2019 course](http://web.stanford.edu/class/cs234/index.html)
2. [Reinforcement Learning: An Introduction - Sutton & Barto](http://incompleteideas.net/book/RLbook2018.pdf)
3. [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
4. [Playing Atari with Deep Reinforcement Learning - Mnih et. al.](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
5. [Deepmind's Nature Paper on Deep Q-Learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
