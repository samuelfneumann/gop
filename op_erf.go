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

// erfOp is the error function
type erfOp struct{}

// newErfOp returns a new erfOp
func newErfOp() G.Op {
	return &erfOp{}
}

// Arity returns the number of elements this operation takes
func (e *erfOp) Arity() int {
	return 1
}

// Type returns the type of the operation
func (e *erfOp) Type() hm.Type {
	// All pointwise unary operations have this type:
	// op :: (Arithable a) => a -> a
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

// Do runs the erf operation
func (e *erfOp) Do(values ...G.Value) (G.Value, error) {
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

// ReturnsPtr indicates whether this Op returns a pointer to its
// output value - which is true.
func (e *erfOp) ReturnsPtr() bool { return true }

// CallsExtern returns whether this Op calls any external functions -
// which is false
func (e *erfOp) CallsExtern() bool { return false }

// OverwritesInput returns the index of the input that this op
// will overwrite - index 0
func (e *erfOp) OverwritesInput() int { return 0 }

// String returns the string representation of the struct
func (e *erfOp) String() string {
	return "Erf"
}

// InferShape returns the output shape as a function of the inputs
func (e *erfOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
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
func (e *erfOp) WriteHash(h hash.Hash) { fmt.Fprint(h, e.String()) }

// Hashcode returns the hash code of the receiver
func (e *erfOp) Hashcode() uint32 { return SimpleHash(e) }

// SymDiff constructs the symbolic derivative of the Erf
func (e *erfOp) SymDiff(inputs G.Nodes, output,
	grad *G.Node) (G.Nodes, error) {
	err := CheckArity(e, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("symDiff: %v", err)
	}

	diffOp := &erfDiffOp{}
	nodes := make(G.Nodes, 1)

	nodes[0], err = G.ApplyOp(diffOp, inputs[0], grad)

	return nodes, err
}

// DiffWRT returns which inputs the operation is differentiable with
// respect to
func (e *erfOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("erf operator only supports one input, got %d "+
			"instead", inputs))
	}
	return []bool{true}
}

// checkInputs returns an error if the input to this Op is invalid
func (e *erfOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(e, len(inputs)); err != nil {
		return err
	}

	_, okF64 := inputs[0].(*G.F64)
	_, okF32 := inputs[0].(*G.F32)
	t, okTensor := inputs[0].(tensor.Tensor)

	if okTensor && len(t.Shape()) <= 0 {
		return fmt.Errorf("tensor does not have any shape")
	}

	if !(okF64 || okF32 || okTensor) {
		return fmt.Errorf("expected input to be a tensor, got %T", inputs[0])
	}

	return nil
}

// erfDiffOp is the derivative of erf
type erfDiffOp struct{}

// Arity returns the number of elements this operation takes
func (e *erfDiffOp) Arity() int { return 2 }

// ReturnsPtr indicates whether this Op returns a pointer to its
// output value - which is true.
func (e *erfDiffOp) ReturnsPtr() bool { return true }

// CallsExtern returns whether this Op calls any external functions -
// which is false
func (e *erfDiffOp) CallsExtern() bool { return false }

// WriteHash writes the hash of the receiver to a hash struct
func (e *erfDiffOp) WriteHash(h hash.Hash) { fmt.Fprint(h, e.String()) }

// Hashcode returns the hash code of the receiver
func (e *erfDiffOp) Hashcode() uint32 { return SimpleHash(e) }

// String returns the string representation of the erfDiffOp
func (e *erfDiffOp) String() string { return "ErfDiff()" }

// InferShape returns the output shape as a function of the inputs
func (e *erfDiffOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(e, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}
	if inputs[0] == nil {
		return nil, fmt.Errorf("inferShape: nil input")
	}

	return inputs[0].(tensor.Shape), nil
}

// Type returns the type of the operation
func (e *erfDiffOp) Type() hm.Type {
	// All pointwise unary operations have this type:
	// op :: (Arithable a) => a -> a
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

// OverwritesInput returns the index of the input that this op
// will overwrite - this op does not overwrite input.
func (e *erfDiffOp) OverwritesInput() int { return -1 }

// checkInputs returns an error if the input to this Op is invalid
func (e *erfDiffOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(e, len(inputs)); err != nil {
		return err
	}

	_, okF64 := inputs[0].(*G.F64)
	_, okF32 := inputs[0].(*G.F32)
	_, okTensor := inputs[0].(tensor.Tensor)
	var okGrad bool
	if okTensor {
		_, okGrad = inputs[1].(tensor.Tensor)
	} else if okF64 {
		_, okGrad = inputs[1].(*G.F64)
	} else if okF32 {
		_, okGrad = inputs[1].(*G.F32)
	}

	if !((okF64 || okF32 || okTensor) && okGrad) {
		return fmt.Errorf("expected input to be a tensor, got %T", inputs[0])
	}

	return nil
}

// Do computes the derivative of the erf
func (e *erfDiffOp) Do(inputs ...G.Value) (G.Value, error) {
	err := e.checkInputs(inputs...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	// If inputs[0] is a scalar type, take the scalar derivative
	switch v := inputs[0].(type) {
	case *G.F64:
		grad := float64(*(inputs[1].(*G.F64)))
		z := float64(*v)
		diff := (2 / math.Sqrt(math.Pi)) * math.Exp(-math.Pow(z, 2))
		return G.NewF64(grad * diff), nil

	case *G.F32:
		grad := float32(*(inputs[1].(*G.F32)))
		z := float32(*v)
		diff := (2 / math32.Sqrt(math.Pi)) * math32.Exp(-math32.Pow(z, 2))
		return G.NewF32(grad * diff), nil
	}

	// inputs[0] is a tensor type
	x := inputs[0].(tensor.Tensor)
	grad := inputs[1].(tensor.Tensor)

	var ret G.Value
	switch x.Dtype() {
	case tensor.Float64:
		ret = e.f64Kernel(x.Shape().Clone(), x, grad)

	case tensor.Float32:
		ret = e.f32Kernel(x.Shape().Clone(), x, grad)
	}

	return ret, nil
}

// f64Kernel computes the derivative on a tensor of dtype float64
func (e *erfDiffOp) f64Kernel(shape tensor.Shape, inputData,
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

// f32Kernel computes the derivative on a tensor of dtype float32
func (e *erfDiffOp) f32Kernel(shape tensor.Shape, inputData,
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
		val = math32.Erf(val.(float32))
	} else {
		return fmt.Errorf("erfTensorAt: invalid data type %T", v)
	}

	// Set the value
	err = v.SetAt(val, coords...)
	if err != nil {
		return fmt.Errorf("erfTensorAt: could not set element "+
			"at %v", coords)

	}
	return nil
}
