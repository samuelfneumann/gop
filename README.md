# gop: Extended G(orgonia) Op(erations)

This repository adds extended operations to Gorgonia. Although Gorgonia
is a great tool for `Go` developers, it lacks many of the features I
need for reinforcement learning research. This `Go` module provides
those operations.

Due to how data is stored in Gorgonia, only symbolic differentiation
is supported in this module. Automatic differentiation is not supported.
Therefore, you must use a `gorgonia.tapeMachine` if you require that
the operations used in this package be differentiable. Note that many
operations, such as `Argsort` are not differentiable anyway.

Furthermore, only tensors of floating-point types are differentiable. If you
have an integer tensor and perform operations on it, you cannot differentiate
that operation (this is also how Gorgonia deals with tensors). This actually
makes sense. Integer tensors are discrete, and operations that work on integer
tensors are discrete operations, and discrete operations are not
differentiable. So, although you can use the operations defined in this package
on integer tensors, beware that such operations are not differentiable.

There are many different integer types, and almost always the "normal" integer
type `int` is used in place of all other integer types. Because of this, any
operations on integer tensors is this package will **always** result in the output
being of type `tensor.Int`, regardless of what the input integer type is. It's
too much work to implement all operations for all floating point types and all
integer types, especially when most integer types are rarely used. If you need
to use an integer tensor, it's probably safest to use a `tensor.Int` type tensor
if you're working with this package, especially if on a 32-bit machine.

Once `Go` implements generics, this module will be updated to work with
any integer type properly.

## Operations

The following is a list of operations implemented:

Operation Name   |   Differentiable?
-----------------|-------------------
Argsort          | No
Error Function   | Yes
Clamp/Clip       | Yes

The following is a list of planned operations that will be implemented soon:

Operation Name   |   Differentiable?
-----------------|-------------------
Repeat           | Yes
Gather           | Yes

## ToDo

* [ ] Repeat

* [ ] Gather
