package gop

import (
	"fmt"
	"hash"
	"math"

	"github.com/chewxy/hm"
	"github.com/chewxy/math32"
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
	err := e.checkInputs(values...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	if values[0] == nil {
		return nil, fmt.Errorf("do: no input")
	}

	value := values[0]

	// Compute erf based on type, overwriting the input
	return computeErf(value)
}

func (e *ErfOp) ReturnsPtr() bool { return true }

func (e *ErfOp) CallsExtern() bool { return false }

func (e *ErfOp) OverwritesInput() int { return 0 }

// String returns the string representation of the struct
func (e *ErfOp) String() string {
	return "Erf"
}

// InferShape returns the output shape as a function of the inputs
func (e *ErfOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(e, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}
	if inputs[0] == nil {
		return nil, fmt.Errorf("inferShape: nil input")
	}

	return inputs[0].(tensor.Shape), nil
}

// WriteHash writes the hash of the receiver to a hash struct
func (e *ErfOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, "Erf()") }

// Hashcode returns the hash code of the receiver
func (e *ErfOp) Hashcode() uint32 { return SimpleHash(e) }

func (e *ErfOp) SymDiff(inputs G.Nodes, output,
	grad *G.Node) (G.Nodes, error) {
	err := CheckArity(e, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("symDiff: %v", err)
	}

	diffOp := &ErfDiffOp{}
	nodes := make(G.Nodes, 1)

	nodes[0], err = G.ApplyOp(diffOp, inputs[0], grad)

	return nodes, err
}

func (e *ErfOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("erf operator only supports one input, got %d "+
			"instead", inputs))
	}
	return []bool{true}
}

// checkInputs returns an error if the input to this Op is invalid
func (e *ErfOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(e, len(inputs)); err != nil {
		return err
	}

	_, okF64 := inputs[0].(*G.F64)
	_, okF32 := inputs[0].(*G.F32)
	_, okTensor := inputs[0].(tensor.Tensor)

	if !(okF64 || okF32 || okTensor) {
		return fmt.Errorf("expected input to be a tensor, got %T", inputs[0])
	}

	return nil
}

type ErfDiffOp struct{}

func (e *ErfDiffOp) Arity() int { return 2 }

func (e *ErfDiffOp) ReturnsPtr() bool { return true }

func (e *ErfDiffOp) CallsExtern() bool { return false }

func (e *ErfDiffOp) WriteHash(h hash.Hash) { fmt.Fprint(h, e.String()) }

func (e *ErfDiffOp) Hashcode() uint32 { return SimpleHash(e) }

func (e *ErfDiffOp) String() string { return "ErfDiff()" }

func (e *ErfDiffOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(e, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}
	if inputs[0] == nil {
		return nil, fmt.Errorf("inferShape: nil input")
	}

	return inputs[0].(tensor.Shape), nil
}

func (e *ErfDiffOp) Type() hm.Type {
	// All pointwise unary operations have this type:
	// op :: (Arithable a) => a -> a
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (e *ErfDiffOp) OverwritesInput() int { return -1 }

// checkInputs returns an error if the input to this Op is invalid
func (e *ErfDiffOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(e, len(inputs)); err != nil {
		return err
	}

	_, okTensor := inputs[0].(tensor.Tensor)
	_, okGrad := inputs[1].(tensor.Tensor)

	if !(okTensor || okGrad) {
		return fmt.Errorf("expected input to be a tensor, got %T", inputs[0])
	}

	return nil
}

func (e *ErfDiffOp) Do(inputs ...G.Value) (G.Value, error) {
	err := e.checkInputs(inputs...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	x := inputs[0].(tensor.Tensor)
	grad := inputs[1].(tensor.Tensor)

	var ret *tensor.Dense
	switch x.Dtype() {
	case tensor.Float64:
		ret = e.f64Kernel(x.Shape().Clone(), x, grad)

	case tensor.Float32:
		ret = e.f32Kernel(x.Shape().Clone(), x, grad)
	}

	return ret, nil
}

func (e *ErfDiffOp) f64Kernel(shape tensor.Shape, inputData,
	gradData tensor.Tensor) *tensor.Dense {
	scale := 2 / math.Sqrt(math.Pi)
	x := inputData.Data().([]float64)
	grad := gradData.Data().([]float64)

	ret := tensor.New(
		tensor.WithShape(shape...),
		tensor.Of(inputData.Dtype()),
	)

	for i, elem := range x {
		newGrad := grad[i] * scale * math.Exp(-math.Pow(elem, 2))
		ret.Set(i, newGrad)
	}

	return ret
}

func (e *ErfDiffOp) f32Kernel(shape tensor.Shape, inputData,
	gradData tensor.Tensor) *tensor.Dense {
	scale := float32(2.0 / math.Sqrt(math.Pi))
	x := inputData.Data().([]float32)
	grad := gradData.Data().([]float32)

	ret := tensor.New(
		tensor.WithShape(shape...),
		tensor.Of(inputData.Dtype()),
	)

	for i, elem := range x {
		exp := -math32.Pow(elem, 2)
		newGrad := grad[i] * scale * math32.Exp(exp)
		ret.Set(i, newGrad)
	}

	return ret
}

// ================================================================
// ================================================================
// ================================================================
// ================================================================
// f32Erf computes the erf on a float32 input value
func f32Erf(val float32) float32 {
	return float32(math.Erf(float64(val)))
}

// computeErf computes the element-wise erf on a value
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

	case tensor.Tensor:
		if len(v.Shape()) == 0 {
			return nil, fmt.Errorf("do: cannot compute erf on empty tensor")
		}

		iter := v.Iterator()
		_, err := iter.Start()
		if err != nil {
			return nil, fmt.Errorf("do: could not start iterator on tensor")
		}

		// Go through each element of the tensor and erf it in place
		for !iter.Done() {
			// Get the coordinates of the element to erf
			coords := iter.Coord()

			// Erf the element in place
			err := erfTensorAt(v, coords)
			if err != nil {
				return nil, fmt.Errorf("do: %v", err)
			}

			// Step the iterator
			_, _, err = iter.NextValid()
			if err != nil {
				return nil, fmt.Errorf("do: could not step iterator")
			}

		}
		// Erf the last element of the tensor
		coords := iter.Coord()
		erfTensorAt(v, coords)

	default:
		return nil, fmt.Errorf("do: unable to compute erf on type %T", v)
	}

	return value, nil
}

// erfTensorAt computes in-place the erf of tensor v at coords
func erfTensorAt(v tensor.Tensor, coords []int) error {
	// Get the value at the next coordinates
	val, err := v.At(coords...)
	if err != nil {
		return fmt.Errorf("erfTensorAt: could not access element "+
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
		return fmt.Errorf("erfTensorAt: could not set element "+
			"at %v", coords)

	}
	return nil
}
