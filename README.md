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

## ToDo

* [ ] Argsort
