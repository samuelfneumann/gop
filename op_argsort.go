package gop

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"github.com/samuelfneumann/top"
)

type argsortOp struct {
	axis int
}

func newArgsortOp(axis int) *argsortOp { return &argsortOp{axis} }

func (a *argsortOp) Arity() int { return 1 }

func (a *argsortOp) Type() hm.Type {
	// All pointwise unary operations have this type:
	// op :: (Arithable a) => a -> a
	any := hm.TypeVariable('a')
	b := G.TensorType{
		Dims: 1,
		Of:   tensor.Int,
	}
	return hm.NewFnType(any, b)
}

func (a *argsortOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(a, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}
	if inputs[0] == nil {
		return nil, fmt.Errorf("inferShape: nil input")
	}

	return inputs[0].(tensor.Shape), nil
}

func (a *argsortOp) ReturnsPtr() bool { return false }

func (a *argsortOp) CallsExtern() bool { return false }

func (a *argsortOp) OverwritesInput() int { return -1 }

func (a *argsortOp) String() string { return "Argsort()" }

// WriteHash writes the hash of the receiver to a hash struct
func (a *argsortOp) WriteHash(h hash.Hash) { fmt.Fprint(h, a.String()) }

// Hashcode returns the hash code of the receiver
func (a *argsortOp) Hashcode() uint32 { return SimpleHash(a) }

func (a *argsortOp) Do(values ...G.Value) (G.Value, error) {
	err := a.checkInputs(values...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	input := values[0].(tensor.Tensor)

	return top.Argsort(input, a.axis)
}

// checkInputs returns an error if the input to this Op is invalid
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
