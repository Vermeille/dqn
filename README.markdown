Reinforcement Learning
======================

The X is controlled by a neural net, and must eat food ('O'). A step that moves the X closer to the
O gives a reward on 1, a step further gives -1.

Deep Q-Network
==============

The algorithm used is the one described by Deep Mind in
[https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf](this paper).

Usage
=====

Train the network with `--no-draw`. It avoids loosing time in drawing the game and outputs total
rewards every 1000 steps.

If you don't want the network to learn, use `--no-learning`.

Finally, when you're satisfied, use `--demo` to see the network playing without learning, and with
a human speed game drawing.

A pretrained network is in brain.net.

![trained net](tty.gif)
