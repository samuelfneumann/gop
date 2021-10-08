package gop

import (
	"fmt"
	"hash"
	"math"

	"github.com/chewxy/hm"
	"github.com/samuelfneumann/math32"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ToDo: figure out the type system and fix Type() methos

// erfinvOp is the error function
type erfinvOp struct{}

// newErfOp returns a new erfinvOp
func newErfinvOp() G.Op {
	return &erfinvOp{}
}

// Arity returns the number of elements this operation takes
func (e *erfinvOp) Arity() int {
	return 1
}

// Type returns the type of the operation
func (e *erfinvOp) Type() hm.Type {
	// All pointwise unary operations have this type:
	// op :: (Arithable a) => a -> a
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

// Do runs the erfinv operation
func (e *erfinvOp) Do(values ...G.Value) (G.Value, error) {
	err := e.checkInputs(values...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	if values[0] == nil {
		return nil, fmt.Errorf("do: no input")
	}

	value := values[0]

	// Compute erfinv based on type
	return computeErfinv(value)
}

// ReturnsPtr indicates whether this Op returns a pointer to its
// output value - which is true.
func (e *erfinvOp) ReturnsPtr() bool { return true }

// CallsExtern returns whether this Op calls any external functions -
// which is false
func (e *erfinvOp) CallsExtern() bool { return false }

// OverwritesInput returns the index of the input that this op
// will overwrite
func (e *erfinvOp) OverwritesInput() int { return -1 }

// String returns the string representation of the struct
func (e *erfinvOp) String() string {
	return "Erf"
}

// InferShape returns the output shape as a function of the inputs
func (e *erfinvOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(e, len(inputs))
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

// WriteHash writes the hash of the receiver to a hash struct
func (e *erfinvOp) WriteHash(h hash.Hash) { fmt.Fprint(h, e.String()) }

// Hashcode returns the hash code of the receiver
func (e *erfinvOp) Hashcode() uint32 { return SimpleHash(e) }

// SymDiff constructs the symbolic derivative of the Erf
func (e *erfinvOp) SymDiff(inputs G.Nodes, output,
	grad *G.Node) (G.Nodes, error) {
	err := CheckArity(e, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("symDiff: %v", err)
	}

	diffOp := &erfinvDiffOp{}
	nodes := make(G.Nodes, 1)

	nodes[0], err = G.ApplyOp(diffOp, inputs[0], grad)

	return nodes, err
}

// DiffWRT returns which inputs the operation is differentiable with
// respect to
func (e *erfinvOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("erfinv operator only supports one input, got %d "+
			"instead", inputs))
	}
	return []bool{true}
}

// checkInputs returns an error if the input to this Op is invalid
func (e *erfinvOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(e, len(inputs)); err != nil {
		return err
	}

	_, okF64 := inputs[0].(*G.F64)
	_, okF32 := inputs[0].(*G.F32)
	t, okTensor := inputs[0].(tensor.Tensor)

	if okTensor && (len(t.Shape()) <= 0 || t.Size() == 0) {
		return fmt.Errorf("tensor does not have any elements")
	}

	if !(okF64 || okF32 || okTensor) {
		return fmt.Errorf("expected input to be a tensor, got %T", inputs[0])
	}

	return nil
}

// erfinvDiffOp is the derivative of erfinv
type erfinvDiffOp struct{}

// Arity returns the number of elements this operation takes
func (e *erfinvDiffOp) Arity() int { return 2 }

// ReturnsPtr indicates whether this Op returns a pointer to its
// output value - which is true.
func (e *erfinvDiffOp) ReturnsPtr() bool { return true }

// CallsExtern returns whether this Op calls any external functions -
// which is false
func (e *erfinvDiffOp) CallsExtern() bool { return false }

// WriteHash writes the hash of the receiver to a hash struct
func (e *erfinvDiffOp) WriteHash(h hash.Hash) { fmt.Fprint(h, e.String()) }

// Hashcode returns the hash code of the receiver
func (e *erfinvDiffOp) Hashcode() uint32 { return SimpleHash(e) }

// String returns the string representation of the erfinvDiffOp
func (e *erfinvDiffOp) String() string { return "ErfDiff()" }

// InferShape returns the output shape as a function of the inputs
func (e *erfinvDiffOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(e, len(inputs))
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

// Type returns the type of the operation
func (e *erfinvDiffOp) Type() hm.Type {
	// All pointwise unary operations have this type:
	// op :: (Arithable a) => a -> a
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a, a)
}

// OverwritesInput returns the index of the input that this op
// will overwrite - this op does not overwrite input.
func (e *erfinvDiffOp) OverwritesInput() int { return -1 }

// checkInputs returns an error if the input to this Op is invalid
func (e *erfinvDiffOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(e, len(inputs)); err != nil {
		return err
	}

	_, okF64 := inputs[0].(*G.F64)
	_, okF32 := inputs[0].(*G.F32)
	t, okTensor := inputs[0].(tensor.Tensor)
	var okGrad bool
	if okTensor {
		_, okGrad = inputs[1].(tensor.Tensor)
		if len(t.Shape()) <= 0 || t.Size() == 0 {
			return fmt.Errorf("tensor does not have any elements")
		}
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

// Do computes the derivative of the erfinv
func (e *erfinvDiffOp) Do(inputs ...G.Value) (G.Value, error) {
	err := e.checkInputs(inputs...)
	if err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	// If inputs[0] is a scalar type, take the scalar derivative
	switch v := inputs[0].(type) {
	case *G.F64:
		grad := float64(*(inputs[1].(*G.F64)))
		z := float64(*v)
		diff := 0.5 * math.Sqrt(math.Pi) * math.Exp(math.Pow(math.Erfinv(z),
			2))
		return G.NewF64(grad * diff), nil

	case *G.F32:
		grad := float32(*(inputs[1].(*G.F32)))
		z := float32(*v)
		diff := 0.5 * math32.Sqrt(math32.Pi) * math32.Exp(
			math32.Pow(math32.Erfinv(z), 2))
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
func (e *erfinvDiffOp) f64Kernel(shape tensor.Shape, inputData,
	gradData tensor.Tensor) *tensor.Dense {
	scale := math.Sqrt(math.Pi) / 2.0
	x := inputData.Data().([]float64)
	grad := gradData.Data().([]float64)

	ret := tensor.New(
		tensor.WithShape(shape...),
		tensor.Of(inputData.Dtype()),
	)

	for i, elem := range x {
		newGrad := grad[i] * scale * math.Exp(math.Pow(math.Erfinv(elem), 2))
		ret.Set(i, newGrad)
	}

	return ret
}

// f32Kernel computes the derivative on a tensor of dtype float32
func (e *erfinvDiffOp) f32Kernel(shape tensor.Shape, inputData,
	gradData tensor.Tensor) *tensor.Dense {
	scale := math32.Sqrt(math32.Pi) / float32(2.0)
	x := inputData.Data().([]float32)
	grad := gradData.Data().([]float32)

	ret := tensor.New(
		tensor.WithShape(shape...),
		tensor.Of(inputData.Dtype()),
	)

	for i, elem := range x {
		exp := math32.Pow(math32.Erfinv(elem), 2)
		newGrad := grad[i] * scale * math32.Exp(exp)
		ret.Set(i, newGrad)
	}

	return ret
}

// computeErf computes the element-wise erfinv on a value
func computeErfinv(value G.Value) (G.Value, error) {
	// Compute erfinv based on type
	switch v := value.(type) {
	case *G.F64:
		return G.NewF64(math.Erfinv(float64(*v))), nil

	case *G.F32:
		val := math32.Erfinv(float32(*v))
		return G.NewF32(val), nil

	case tensor.Tensor:
		if len(v.Shape()) == 0 {
			return nil, fmt.Errorf("do: cannot compute erfinv on empty tensor")
		}

		// Create the new output tensor
		out := tensor.NewDense(
			v.Dtype(),
			v.Shape(),
		)

		iter := v.Iterator()
		// Go through each element of the tensor and erfinv it in place
		for !iter.Done() {
			// Get the coordinates of the element to erfinv
			coords := iter.Coord()

			// Erfinv v, storing results in out
			err := erfinvTensorAt(v, out, coords)
			if err != nil {
				return nil, fmt.Errorf("do: %v", err)
			}

			// Step the iterator
			_, _, err = iter.NextValid()
			if err != nil {
				return nil, fmt.Errorf("do: could not step iterator")
			}
		}

		return out, nil

	default:
		return nil, fmt.Errorf("do: unable to compute erfinv on type %T", v)
	}
}

// erfinvTensorAt computes erfinv of tensor v at coords, storing the
// result in out at coords
func erfinvTensorAt(in tensor.Tensor, out tensor.Tensor, coords []int) error {
	// Get the value at the next coordinates
	val, err := in.At(coords...)
	if err != nil {
		return fmt.Errorf("erfinvTensorAt: could not access element "+
			"at %v", coords)
	}

	// Erf the value
	if in.Dtype() == tensor.Float64 {
		val = math.Erfinv(val.(float64))
	} else if in.Dtype() == tensor.Float32 {
		val = math32.Erfinv(val.(float32))
	} else {
		return fmt.Errorf("erfinvTensorAt: invalid data type %v", in.Dtype())
	}

	// Set the value
	err = out.SetAt(val, coords...)
	if err != nil {
		return fmt.Errorf("erfinvTensorAt: could not set element "+
			"at %v", coords)

	}
	return nil
}
