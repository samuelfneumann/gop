package gop

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// repeatOp implements the repeat operation. This operation is the
// same operation as Numpy's repeat operation and PyTorch's
// repeat_interleave operation. It repeats elements of an array.
type repeatOp struct {
	axis    int // Axis along which to repeat
	dims    int // Number of dimensions in the input node
	repeats int // Number of times each row is repeated
}

// newRepeatOp returns a new repeatOp
func newRepeatOp(axis, dims, repeats int) (*repeatOp, error) {
	if repeats == 0 {
		return nil, fmt.Errorf("newRepeatOp: expected repeats to be > 0, "+
			"got %v", repeats)
	}

	return &repeatOp{
		axis:    axis,
		dims:    dims,
		repeats: repeats,
	}, nil
}

// DiffWRT implements the gorgonia.SDOp interface
func (r *repeatOp) DiffWRT(inputs int) []bool {
	return []bool{true}
}

// SymDiff implements the gorgonia.SDOp interface
func (r *repeatOp) SymDiff(inputs G.Nodes, output, grad *G.Node) (G.Nodes,
	error) {
	err := CheckArity(r, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("symDiff: %v", err)
	}

	diffOp := &repeatDiffOp{r}
	nodes := make(G.Nodes, 1)

	nodes[0], err = G.ApplyOp(diffOp, inputs[0], grad)

	return nodes, err
}

// Arity implements the gorgonia.Op interface
func (r *repeatOp) Arity() int { return 1 }

// Type implements the gorgonia.Op interface
func (r *repeatOp) Type() hm.Type {
	tt := G.TensorType{
		Dims: r.dims,
		Of:   hm.TypeVariable('a'),
	}

	return hm.NewFnType(tt, tt)
}

// OverwritesInput implements the gorgonia.Op interface
func (r *repeatOp) OverwritesInput() int { return -1 }

// ReturnsPtr implements the gorgonia.Op interface
func (r *repeatOp) ReturnsPtr() bool { return true }

// CallsExtern implements the gorgonia.Op interface
func (r *repeatOp) CallsExtern() bool { return false }

// String implements the fmt.Stringer interface
func (r *repeatOp) String() string {
	return fmt.Sprintf("Repeat{axis=%v, repeats=%v}()", r.axis, r.repeats)
}

// WriteHash implements the gorgonia.Op interface
func (r *repeatOp) WriteHash(h hash.Hash) { fmt.Fprint(h, r.String()) }

// Hashcode implements the gorgonia.Op interface
func (r *repeatOp) Hashcode() uint32 { return SimpleHash(r) }

// InferShape implements the gorgonia.Op interface
func (r *repeatOp) InferShape(in ...G.DimSizer) (tensor.Shape, error) {
	shape := in[0].(tensor.Shape).Clone()
	shape[r.axis] *= r.repeats

	return shape, nil
}

// Do implements the gorgonia.Op interface
func (r *repeatOp) Do(inputs ...G.Value) (G.Value, error) {
	if err := r.checkInputs(inputs...); err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	input := inputs[0].(tensor.Tensor)

	return tensor.Repeat(input, r.axis, r.repeats)
}

// checkInputs returns an error if inputs is not a valid input to the
// receiver
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

// repeatDiffOp is the derivative of the repeat operation
type repeatDiffOp struct {
	op *repeatOp
}

// Arity implements the gorgonia.Op interface
func (r *repeatDiffOp) Arity() int { return 2 }

// Type implements the gorgonia.Op interface
func (r *repeatDiffOp) Type() hm.Type {
	tt := G.TensorType{
		Dims: r.op.dims,
		Of:   hm.TypeVariable('a'),
	}

	return hm.NewFnType(tt, tt, tt)
}

// OverwritesInput implements the gorgonia.Op interface
func (r *repeatDiffOp) OverwritesInput() int { return -1 }

// ReturnsPtr implements the gorgonia.Op interface
func (r *repeatDiffOp) ReturnsPtr() bool { return true }

// CallsExtern implements the gorgonia.Op interface
func (r *repeatDiffOp) CallsExtern() bool { return false }

// String implements the gorgonia.Op interface
func (r *repeatDiffOp) String() string {
	return fmt.Sprintf("RepeatDiff{axis=%v, repeats=%v}()", r.op.axis,
		r.op.repeats)
}

// WriteHash implements the gorgonia.Op interface
func (r *repeatDiffOp) WriteHash(h hash.Hash) { fmt.Fprint(h, r.String()) }

// Hashcode implements the gorgonia.Op interface
func (r *repeatDiffOp) Hashcode() uint32 { return SimpleHash(r) }

// InferShape implements the gorgonia.Op interface
func (r *repeatDiffOp) InferShape(in ...G.DimSizer) (tensor.Shape, error) {
	shape := in[0].(tensor.Shape).Clone()
	shape[r.op.axis] /= r.op.repeats

	return shape, nil
}

// Do implements the gorgonia.Op interface
func (r *repeatDiffOp) Do(inputs ...G.Value) (G.Value, error) {
	if err := r.checkInputs(inputs...); err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	grad := inputs[1].(tensor.Tensor)
	shape, err := r.InferShape(grad.Shape())
	if err != nil {
		return nil, fmt.Errorf("do: could not infer shape: %v", err)
	}

	if r.op.repeats == 1 {
		return grad.Clone().(*tensor.Dense), nil
	}

	outRows := make([]*tensor.Dense, shape[r.op.axis])

	slices := make([]tensor.Slice, len(shape))
	for i := 0; i < shape[r.op.axis]; i++ {
		start := i * r.op.repeats
		stop := start + r.op.repeats
		slices[r.op.axis] = G.S(start, stop)

		slice, err := grad.Slice(slices...)
		if err != nil {
			return nil, fmt.Errorf("do: could not slice grad: %v", err)
		}

		var outRow *tensor.Dense
		if slice.Size() > 1 {
			outRow, err = slice.(*tensor.Dense).Sum(r.op.axis)
		} else {
			outRow = slice.(*tensor.Dense)
		}
		if err != nil {
			return nil, fmt.Errorf("do: could not sum grad: %v", err)
		}
		outRows[i] = outRow
	}

	if shape[r.op.axis] > 1 {
		return outRows[0].Stack(r.op.axis, outRows[1:]...)
	}
	return outRows[0], nil
}

// checkInputs returns an error if inputs is not a valid input to the
// receiver
func (r *repeatDiffOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(r, len(inputs)); err != nil {
		return err
	}

	t, ok := inputs[1].(tensor.Tensor)
	if !ok {
		return fmt.Errorf("expected tensor, received %T", inputs[0])
	} else if t == nil {
		return fmt.Errorf("cannot diff repeat nil tensor")
	} else if t.Size() == 0 {
		return fmt.Errorf("cannot diff repeat empty tensor")
	} else if r.op.axis >= len(t.Shape()) {
		return fmt.Errorf("axis [%v] out of range for tensor with shape %v",
			r.op.axis, t.Shape())
	}

	return nil
}
