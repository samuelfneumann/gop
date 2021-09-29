package gop

import G "gorgonia.org/gorgonia"

func Erf(x *G.Node) (*G.Node, error) {
	op := NewErfOp()

	return G.ApplyOp(op, x)
}
