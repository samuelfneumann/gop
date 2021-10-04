# gop: Extended G(orgonia) Op(erations)

This repository adds extended operations to Gorgonia. Although Gorgonia
is a great tool for `Go` developers, it lacks many of the features I
need for reinforcement learning research. This `Go` module provides
those operations that my research needs.

Due to how data is stored in Gorgonia, only symbolic differentiation
is supported in this module. Automatic differentiation is not supported.
Therefore, you must use a `gorgonia.tapeMachine` if you require that
the operations used in this package be differentiable. Note that many
operations, such as `Argsort` are not differentiable anyway.

## Operations

All operations in this module are meant to be used with either `float64`
or `float32` tensors, although many operations also support usage on
integer tensor types. Note that differentiation is only supported for
floating point tensor types in this module (as well as in Gorgonia).
For example if `in` is a tensor of type `tensor.Float64`, then
the gradient of `Clamp(in, -1.0, 1.0, false)` exists. If `in` is
a tensor of type `tensor.Int`, the gradient does not exist.

The following is a list of operations implemented:

Operation Name   |   Differentiable?
-----------------|-------------------
Argsort          | No
Error Function   | Yes
Clamp/Clip       | Yes

## ToDo

* [ ] Repeat

* [ ] Gather
