package op

import (
	"fmt"
	"hash"
	"math"

	"github.com/chewxy/hm"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ErfOp is the error function
type ErfOp struct{}

func NewErfOp() G.Op {
	return &ErfOp{}
}

func (e *ErfOp) Arity() int {
	return 1
}

func (e *ErfOp) Type() hm.Type {
	// All pointwise unary operations have this type:
	// op :: (Arithable a) => a -> a
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (e *ErfOp) Do(values ...G.Value) (G.Value, error) {
	err := CheckArity(e, len(values))
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	if len(values) != 1 {
		return nil, fmt.Errorf("do: expected 1 value, got %v", len(values))
	}

	if values[0] == nil {
		return nil, fmt.Errorf("do: no input")
	}

	value := values[0]

	// Compute erf based on type, overwriting the input
	return computeErf(value)
}

func (e *ErfOp) ReturnsPtr() bool {
	return true
}

func (e *ErfOp) CallsExtern() bool {
	return false
}

func (e *ErfOp) OverwritesInput() int {
	return 0
}

func (e *ErfOp) String() string {
	return "Erf"
}

func (e *ErfOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, fmt.Errorf("inferShape: nil input")
	}

	return inputs[0].(tensor.Shape), nil
}

func (e *ErfOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Erf()")
}

func (e *ErfOp) Hashcode() uint32 {
	return SimpleHash(e)
}

func prod(ints ...int) int {
	total := 1
	for _, i := range ints {
		total *= i
	}
	return total
}

func f32Erf(val float32) float32 {
	return float32(math.Erf(float64(val)))
}

func computeErf(value G.Value) (G.Value, error) {
	// Compute erf based on type, overwriting the input
	switch v := value.(type) {
	case *G.F64:
		*v = *G.NewF64(math.Erf(float64(*v)))
		return v, nil

	case *G.F32:
		val := float32(math.Erf(float64(*v)))
		*v = *G.NewF32(val)
		return v, nil

	case *tensor.Dense:
		if v.Dtype() == tensor.Float64 {
			for i := 0; i < prod(v.Shape()...); i++ {
				v.Set(i, math.Erf(v.Get(i).(float64)))
			}
		} else if v.Dtype() == tensor.Float32 {
			for i := 0; i < prod(v.Shape()...); i++ {
				v.Set(i, f32Erf(v.Get(i).(float32)))
			}
		} else {
			return nil, fmt.Errorf("do: invalid tensor type %v", v.Dtype())
		}

	case tensor.Tensor:
		iter := v.Iterator()
		_, err := iter.Start()
		if err != nil {
			return nil, fmt.Errorf("do: could not start iterator on tensor")
		}

		coords := iter.Coord()

		for !iter.Done() {
			// Get the value at the next coordinates
			val, err := v.At(coords...)
			if err != nil {
				return nil, fmt.Errorf("do: could not access element "+
					"at %v", coords)
			}

			// Erf the value
			if v.Dtype() == tensor.Float64 {
				val = math.Erf(val.(float64))
			} else if v.Dtype() == tensor.Float32 {
				val = f32Erf(val.(float32))
			}

			// Set the value
			err = v.SetAt(val, coords...)
			if err != nil {
				return nil, fmt.Errorf("do: could not set element "+
					"at %v", coords)

			}

			// Step the iterator
			_, _, err = iter.NextValid()
			if err != nil {
				return nil, fmt.Errorf("do: could not step iterator")
			}
			coords = iter.Coord()
		}

	default:
		return nil, fmt.Errorf("do: unable to compute erf on type %T", v)
	}

	return value, nil
}
