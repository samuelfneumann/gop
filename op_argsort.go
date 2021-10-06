package gop

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"github.com/samuelfneumann/top"
)

// argsortOp is the argsort operation
type argsortOp struct {
	axis int
	dims int // The number of dimensions in the input tensor to argsort
}

// newArgsortOp returns a new argsortOp
func newArgsortOp(axis int, dims int) *argsortOp {
	return &argsortOp{
		axis: axis,
		dims: dims,
	}
}

// Arity implements the gorgonia.Op interface
func (a *argsortOp) Arity() int { return 1 }

// Type implements the gorgonia.Op interface
func (a *argsortOp) Type() hm.Type {
	any := hm.TypeVariable('a')
	b := G.TensorType{
		Dims: a.dims,
		Of:   tensor.Int,
	}
	return hm.NewFnType(any, b)
}

// InferShape implements the gorgonia.Op interface
func (a *argsortOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(a, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}
	if inputs[0] == nil {
		return nil, fmt.Errorf("inferShape: nil input")
	}

	shapes, err := G.DimSizersToShapes(inputs)
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}
	return shapes[0], nil
}

// ReturnsPtr implements the gorgonia.Op interface
func (a *argsortOp) ReturnsPtr() bool { return false }

// CallsExtern implements the gorgonia.Op interface
func (a *argsortOp) CallsExtern() bool { return false }

// OverwriteInput implements the gorgonia.Op interface
func (a *argsortOp) OverwritesInput() int { return -1 }

// String implements the fmt.Stringer interface
func (a *argsortOp) String() string { return "Argsort()" }

// WriteHash implements the gorgonia.Op interface
func (a *argsortOp) WriteHash(h hash.Hash) { fmt.Fprint(h, a.String()) }

// Hashcode implements the gorgonia.Op interface
func (a *argsortOp) Hashcode() uint32 { return SimpleHash(a) }

// Do implements the gorgonia.Op interface
func (a *argsortOp) Do(values ...G.Value) (G.Value, error) {
	err := a.checkInputs(values...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	input := values[0].(tensor.Tensor)

	return top.Argsort(input, a.axis)
}

// checkInputs returns an error if the input to the receiver is invalid
func (a *argsortOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(a, len(inputs)); err != nil {
		return err
	}

	t, ok := inputs[0].(tensor.Tensor)

	if !ok {
		return fmt.Errorf("expected input to be a tensor, got %T", inputs[0])
	}

	if len(t.Shape()) <= 0 || t.Size() == 0 {
		return fmt.Errorf("tensor does not have any elements")
	}

	if len(t.Shape()) <= a.axis {
		return fmt.Errorf("axis out of range [%v] with tensor shape %v",
			a.axis, t.Shape())
	}

	return nil
}
