package gop

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type repeatOp struct {
	axis    int
	repeats int
}

func newRepeatOp(axis int, repeats int) (*repeatOp, error) {
	if repeats == 0 {
		return nil, fmt.Errorf("newRepeatOp: expected repeats to be > 0, "+
			"got %v", repeats)
	}

	return &repeatOp{
		axis:    axis,
		repeats: repeats,
	}, nil
}

func (r *repeatOp) Arity() int { return 1 }

func (r *repeatOp) Type() hm.Type {
	panic("not implemented")
}

func (r *repeatOp) OverwritesInput() int { return -1 }

func (r *repeatOp) ReturnsPtr() bool { return true }

func (r *repeatOp) CallsExtern() bool { return false }

func (r *repeatOp) String() string {
	return fmt.Sprintf("Repeat{axis=%v, repeats=%v}()", r.axis, r.repeats)
}

func (r *repeatOp) WriteHash(h hash.Hash) { fmt.Fprint(h, r.String()) }

func (r *repeatOp) Hashcode() uint32 { return SimpleHash(r) }

func (r *repeatOp) InferShape(in ...G.DimSizer) (tensor.Shape, error) {
	shape := in[0].(tensor.Shape)
	shape[r.axis] *= r.repeats

	return shape, nil
}

func (r *repeatOp) Do(inputs ...G.Value) (G.Value, error) {
	if err := r.checkInputs(inputs...); err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	input := inputs[0].(tensor.Tensor)

	return tensor.Repeat(input, r.axis, r.repeats)
}

func (r *repeatOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(r, len(inputs)); err != nil {
		return err
	}

	t, ok := inputs[0].(tensor.Tensor)
	if !ok {
		return fmt.Errorf("expected tensor, received %T", inputs[0])
	} else if t == nil {
		return fmt.Errorf("cannot repeat nil tensor")
	} else if t.Size() == 0 {
		return fmt.Errorf("cannot repeat empty tensor")
	} else if r.axis >= len(t.Shape()) {
		return fmt.Errorf("axis [%v] out of range for tensor with shape %v",
			r.axis, t.Shape())
	}

	return nil
}
