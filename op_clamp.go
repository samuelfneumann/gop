package gop

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"github.com/samuelfneumann/top"
)

// clampOp implements the clamp operation, clamping all values in a
// tensor to be within some range.
type clampOp struct {
	min, max     interface{}
	passGradient bool
}

// newClamp returns a new clampOp
func newClampOp(min, max interface{}, passGradient bool) (*clampOp, error) {
	op := &clampOp{
		min:          min,
		max:          max,
		passGradient: passGradient,
	}

	return op, nil
}

// DiffWRT implements the gorgonia.SDOp interface
func (c *clampOp) DiffWRT(inputs int) []bool {
	return []bool{true}
}

// SymDiff implements the gorgonia.SDOp interface
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

// Arity implements the gorgonia.Op interface
func (c *clampOp) Arity() int { return 1 }

// Type implements the gorgonia.Op interface
func (c *clampOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a)
}

// InferShape implements the gorgonia.Op interface
func (c *clampOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(c, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}

	shapes, err := G.DimSizersToShapes(inputs)
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}
	return shapes[0], nil
}

// ReturnsPtr implements the gorgonia.Op interface
func (c *clampOp) ReturnsPtr() bool { return true }

// CallsExtern implements the gorgonia.Op interface
func (c *clampOp) CallsExtern() bool { return false }

// OverwritesInput implements the gorgonia.Op interface
func (c *clampOp) OverwritesInput() int { return -1 }

// String implements the fmt.Stringer interface
func (c *clampOp) String() string { return "Clamp() " }

// WriteHash implements the gorgonia.Op interface
func (c *clampOp) WriteHash(h hash.Hash) { fmt.Fprint(h, c.String()) }

// Hashcode implements the gorgonia.Op interface
func (c *clampOp) Hashcode() uint32 { return SimpleHash(c) }

// Do implements the gorgonia.Op interface
func (c *clampOp) Do(inputs ...G.Value) (G.Value, error) {
	if err := c.checkInputs(inputs...); err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	in := inputs[0].(tensor.Tensor)

	cl, err := tensor.Clamp(in, c.min, c.max)

	return cl, err
}

// checkInputs returns an error if inputs is an invalid input for
// clampOp
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

// clampDiffOp is the gradient of clampOp
type clampDiffOp struct {
	op *clampOp
}

// Arity implements the gorgonia.Op interface
func (c *clampDiffOp) Arity() int { return 2 }

// Type implements the gorgonia.Op interface
func (c *clampDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a, a)
}

// InferShape implements the gorgonia.Op interface
func (c *clampDiffOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(c, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}

	shapes, err := G.DimSizersToShapes(inputs)
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}

	return shapes[0], nil
}

// RteurnsPtr implements the gorgonia.Op interface
func (c *clampDiffOp) ReturnsPtr() bool { return false }

// CallsExtern implements the gorgonia.Op interface
func (c *clampDiffOp) CallsExtern() bool { return false }

// OverwritesInput implements the gorgonia.Op interface
func (c *clampDiffOp) OverwritesInput() int { return -1 }

// WriteHash implements the gorgonia.Op interface
func (c *clampDiffOp) WriteHash(h hash.Hash) { fmt.Fprint(h, c.String()) }

// Hashcode implements the gorgonia.Op interface
func (c *clampDiffOp) Hashcode() uint32 { return SimpleHash(c) }

// String implements the fmt.Stringer interface
func (c *clampDiffOp) String() string { return "ClampDiff()" }

// Do implements the gorgonia.Op interface
func (c *clampDiffOp) Do(inputs ...G.Value) (G.Value, error) {
	err := c.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	x := inputs[0].(tensor.Tensor)
	dzdy := inputs[1].(tensor.Tensor)

	var dydx G.Value
	if !c.op.passGradient {
		dydx, err = top.ClampB(x, c.op.min, c.op.max)
	} else {
		dydx, err = tensor.Ones(x.Dtype(), x.Shape()...), nil
	}
	if err != nil {
		return nil, fmt.Errorf("do: could not clampb: %v", err)
	}

	return tensor.Mul(dzdy, dydx)
}

// checkInputs returns an error if inputs in an invalid input to
// clampDiffOp
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
