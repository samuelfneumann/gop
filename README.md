# gop: Extended G(orgonia) Op(erations)

This repository adds extended operations to MLPs in Gorgonia.
Although Gorgonia is a great tool for `Go` developers, it lacks many of
the features I need for reinforcement learning research. This `Go` module
provides those operations.

Due to how I work with Gorgonia, only symbolic differentiation
is supported in this module as of yet. Automatic differentiation is not
supported. Note that many operations, such as `Argsort` are not
differentiable anyway. Eventually, AutoDiff will be supported.

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

Operation Name           |   SymDiff?   |   AutoDiff?
-------------------------|--------------|--------------
Argsort                  | No           | No
Error Function           | Yes          | No
Inverse Error Function   | Yes          | No
Clamp/Clip               | Yes          | No
Repeat                   | Yes          | No
Gather                   | In progress  | No
NormalSample             | No           | No
ReduceMean               | Yes          | Yes
ReduceAdd                | Yes          | Yes
ReduceSub                | Yes          | Yes
ReduceProd               | Yes          | Yes
ReduceDiv                | Yes          | Yes
Squeeze                  | Yes          | Yes
Unsqueeze                | Yes          | Yes
SqueezeAll               | Yes          | Yes
SqueezeAllBut            | Yes          | Yes

## Distributions

The following is a list of probability distributions implemented:

* Univariate Normal

## ToDo

* [ ] Permute/RollAxis

* [ ] StopGradient

* [ ] `distributions.Independent`, similar to PyTorch's `Independent`