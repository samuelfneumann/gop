package gop

import G "gorgonia.org/gorgonia"

func Erf(x *G.Node) (*G.Node, error) {
	op := newErfOp()

	return G.ApplyOp(op, x)
}
