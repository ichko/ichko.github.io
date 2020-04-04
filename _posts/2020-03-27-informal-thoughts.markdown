---
layout: post
title: Informal thoughts related to neural architectures
date: 2020-03-27 13:38:03 +0200
categories:
---

## Transition representations

_01.04.2020_

An attempt at parallelizing recurrent nets during training. This is just and idea,
which I have not had the time to test yet.
If there is work already done on something similar please let me know.

Setup -- let's say you have a set of sequential data:

$$
  D =  \{(X_i, Y_i) \}_{i=0}^{N} \\
  X_i = (x_0, x_1, ..., x_j) \\
  Y_i = (y_0, y_1, ..., y_j)
$$

and you want to train a neural network to uncover the structure in the sequences -- predict
input from output as an example.
There are different ways, depending on the nature of the sequences, that can help you
model that, but one of the most universal neural network architectures one could use is
some sort of recurrent neural network -- **RNN**.

RNNs alow you to model data with arbitrary sequence size, which is what makes them
special. They are characterized with state vector that changes during the processing of
every input and stores anything the model deems important.

$$
  f(s_t, x_t) = [s_{t+1}, y_t]
$$

You give the network state ($s_t$) and input ($x_t$) and it spits out the next state
($s_{t+1}$) and output ($y_t$).
This looks really natural and generic. The task of the network during training is to
learn to **remember** the important parts of the input and to use them when it they are needed.

---

The problem of this setup is that the process of training is **sequential by nature**.
If you want to compute the output for time step $t$, you need to know
the state of the network at time step $t-1$. This recursively shows us that
to do an optimization step, one should process forward and then backward
the whole sequence, sequentially.

> The biggest lesson that can be read from 70 years of AI research is that
> general methods that leverage computation are ultimately the most effective,
> and by a large margin.
>
> Rich Sutton -- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

I've been thinking a lot about this lately and how can we scale it up
and I came up with and interesting, by my biased opinion, way of training every
time step in parallel. That is to say, every time step can be processed
forward and backward independently.

How can one do that if the input at every time step is dependent on the output of
the previous time step. Well, lets say we have that output vector, already computed
and stored somewhere. In a way the computed state becomes trainable, embedding, parameter.

$$
  (x_i, y_i) - an\ arbitrary\ transition \\
  f(e_i, x_i) = [e_{i+i}, y_i]
$$

This can easily be done by associating every time transition, of every sequence,
with two unique ids and then, during training, we can associate the input and output
states, corresponding to these ids with **embedding
vectors** from an embedding layer -- hence representation of the transition.
Then the gradient signal from backprop can update these vectors, refining them as needed.

The training procedure becomes more memory heavy, but also fully parallel.
Which is what dynamic programming algorithms are known for --
trading speed for the price of memory.

The main problem I see with this is having **nonstationary** inputs and outputs. Every
optimization step changes the inputs and the outputs -- the state vectors.
So the optimization procedure will be trying to hit a moving target.
This can make training unstable, as seen in DQN, but there are ways which can
help with this problem.

The thing that comes to mind is what the DQN optimization procedure does -- which is to
update the target network less frequently. This would mean that we should freeze the embeddings
for a few optimization steps, making the inputs and the outputs stationary for a while and
accumulating the gradients for the embedding layer somewhere else, and then update.
The hope is that this would stabilize the procedure and make the whole process converge faster.

In conclusion, I think the described method is quite simple, implementation-wise,
and really promising and I hope I can try it out in the neat future.

<!-- ## Parallel sequence modeling

How can we train fully time conditional recurrent model in parallel, since every output should be conditioned
on what happened before it?
We can use the fact that during training time we have dataset with full rollouts of what we are modelling.
Actually the idea that I am suggesting is independent of that. We can just have input output pairs.
Where are we going to encode the time dependent state? In the embedding of the input output pair.

- We start by generating embedding vectors (we can thing of them as representative of the hidden RNN state)
  for each in-out pair.
- During training we backprop the gradient tuning the embeddings.
  - This allows the time dependent state to be encoded in the embedding.
  - If we don't do that we can have IO pairs with equal input and different outputs, because
    in the real world rollouts outputs would also depend on other, timely factors,
    and this would lead to the network, learning to predict the superpositions of
    all outputs related to single input (blurry predictions).
  - With the embedding vectors we allow the network to encode this variation in the embedding itself. -->

<!-- ## Smooth time modeling

- Project time in multiple dimensions - Each component characterizes the degree in which time influences
  different geometric dimensions.
- Multi scale seasonality (mod or sine) projected to `time embedding vector` -->
