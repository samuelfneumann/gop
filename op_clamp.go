package gop

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"github.com/samuelfneumann/top"
)

type clampOp struct {
	min, max     interface{}
	passGradient bool
}

func newClamp(min, max interface{}, passGradient bool) (*clampOp, error) {
	op := &clampOp{
		min:          min,
		max:          max,
		passGradient: passGradient,
	}

	return op, nil
}

func (c *clampOp) DiffWRT(inputs int) []bool {
	return []bool{true}
}

func (c *clampOp) SymDiff(inputs G.Nodes, output, grad *G.Node) (G.Nodes,
	error) {
	err := CheckArity(c, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("symDiff: %v", err)
	}

	diffOp := &clampDiffOp{c}
	nodes := make(G.Nodes, 1)

	nodes[0], err = G.ApplyOp(diffOp, inputs[0], grad)

	return nodes, err
}

func (c *clampOp) Arity() int { return 1 }

func (c *clampOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a)
}

func (c *clampOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape), nil
}

func (c *clampOp) ReturnsPtr() bool { return true }

func (c *clampOp) CallsExtern() bool { return false }

func (c *clampOp) OverwritesInput() int { return -1 }

func (c *clampOp) String() string { return "Clamp() " }

// WriteHash writes the hash of the receiver to a hash struct
func (c *clampOp) WriteHash(h hash.Hash) { fmt.Fprint(h, c.String()) }

// Hashcode returns the hash code of the receiver
func (c *clampOp) Hashcode() uint32 { return SimpleHash(c) }

func (c *clampOp) Do(inputs ...G.Value) (G.Value, error) {
	if err := c.checkInputs(inputs...); err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	in := inputs[0].(tensor.Tensor)

	cl, err := tensor.Clamp(in, c.min, c.max)

	return cl, err
}

func (c *clampOp) checkInputs(inputs ...G.Value) error {
	err := CheckArity(c, len(inputs))
	if err != nil {
		return err
	}

	t, okTensor := inputs[0].(tensor.Tensor)

	if !okTensor {
		return fmt.Errorf("expected a tensor to clamp but got %T", inputs[0])
	} else if inputs[0] == nil {
		return fmt.Errorf("cannot clamp nil tensor")
	} else if t.Size() == 0 {
		return fmt.Errorf("tensor must have more than 1 row per "+
			"dimension but got shape %v", t.Shape())
	}

	return nil
}

type clampDiffOp struct {
	op *clampOp
}

func (c *clampDiffOp) Arity() int { return 2 }

func (c *clampDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a, a)
}

func (c *clampDiffOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape), nil
}

func (c *clampDiffOp) ReturnsPtr() bool { return false }

func (c *clampDiffOp) CallsExtern() bool { return false }

func (c *clampDiffOp) OverwritesInput() int { return -1 }

// WriteHash writes the hash of the receiver to a hash struct
func (c *clampDiffOp) WriteHash(h hash.Hash) { fmt.Fprint(h, c.String()) }

// Hashcode returns the hash code of the receiver
func (c *clampDiffOp) Hashcode() uint32 { return SimpleHash(c) }

func (c *clampDiffOp) String() string { return "ClampDiff()" }

func (c *clampDiffOp) Do(inputs ...G.Value) (G.Value, error) {
	err := c.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	in := inputs[0].(tensor.Tensor)

	if !c.op.passGradient {
		return top.ClampB(in, c.op.min, c.op.max)
	} else {
		return tensor.Ones(in.Dtype(), in.Shape()...), nil
	}
}

func (c *clampDiffOp) checkInput(inputs ...G.Value) error {
	err := CheckArity(c, len(inputs))
	if err != nil {
		return err
	}

	t, okTensor := inputs[0].(tensor.Tensor)

	if !okTensor {
		return fmt.Errorf("expected a tensor to clamp but got %T", inputs[0])
	} else if inputs[0] == nil {
		return fmt.Errorf("cannot clamp nil tensor")
	} else if t.Size() == 0 {
		return fmt.Errorf("tensor must have more than 1 row per "+
			"dimension but got shape %v", t.Shape())
	}

	return nil
}
