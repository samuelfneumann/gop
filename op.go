// Package op provides extended operations for Gorgonia
package gop

import (
	"fmt"
	"os"
	"runtime"
	"time"

	colour "github.com/samuelfneumann/gocolour"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	GoArch64Bit := []string{"amd64", "arm64", "arm64be", "loong64",
		"mips64", "mips64le", "ppc64", "ppc64le", "riscv64",
		"s390x", "sparc64", "wasm",
	}

	flag := false
	for _, arch := range GoArch64Bit {
		if arch == runtime.GOARCH {
			flag = true
			break
		}
	}

	if !flag {
		fmt.Fprintf(os.Stderr,
			colour.Red+"WARNING: using 32-bit precision, use caution when "+
				"using top with int64 values which may be cast to int "+
				"(int32)"+colour.Reset)
	}
}

func Gather(x *G.Node, axis int, indices *G.Node) (*G.Node, error) {
	op, err := newGatherOp(axis, indices.Shape().Dims())
	if err != nil {
		return nil, fmt.Errorf("gather: %v", err)
	}

	return G.ApplyOp(op, x, indices)
}

// Squeeze removes an axis if it has a length of 1, otherwise an error is
// returned
func Squeeze(x *G.Node, axis int) (*G.Node, error) {
	if x.Shape()[axis] != 1 {
		return x, nil
	}

	shape := x.Shape()[:axis]
	if axis < len(x.Shape())-1 {
		shape = append(shape, x.Shape()[axis+1:]...)
	}

	out, err := G.Reshape(x, shape)
	if err != nil {
		return nil, fmt.Errorf("squeeze: %v", err)
	}

	return out, err
}

// SqueezeAll squeezes all dimensions
func SqueezeAll(x *G.Node) (*G.Node, error) {
	return SqueezeAllBut(x, -1)
}

// SqueezeAllBut squeezes all dimensions but axis
func SqueezeAllBut(x *G.Node, axis int) (*G.Node, error) {
	var err error
	shape := x.Shape()
	dimToSqueeze := 0
	for {
		if shape[dimToSqueeze] == 1 && dimToSqueeze != axis {
			fmt.Println(x.Shape(), dimToSqueeze, shape[dimToSqueeze], axis)
			time.Sleep(time.Second)
			x, err = Squeeze(x, dimToSqueeze)
			if err != nil {
				return nil, fmt.Errorf("reduceAdd: could not squeeze dim %v",
					dimToSqueeze)
			}
			if dimToSqueeze < axis {
				axis--
			}
			shape = x.Shape()
		} else {
			dimToSqueeze++
		}

		if dimToSqueeze >= x.Dims() {
			break
		}
	}

	return x, nil
}

// ReduceMean calculates the mean along axis and squeezes the axis out
func ReduceMean(x *G.Node, axis int) (*G.Node, error) {
	length := x.Shape()[axis]

	sum, err := ReduceAdd(x, axis)
	if err != nil {
		return nil, fmt.Errorf("reduceMean: could not sum: %v", err)
	}

	var n *G.Node
	if x.Dtype() == tensor.Float64 {
		n = G.NewConstant(float64(length))
	} else if x.Dtype() == tensor.Float32 {
		n = G.NewConstant(float32(length))
	} else {
		return nil, fmt.Errorf("reduceMean: cannot compute mean of tensor "+
			"with type %v", x.Dtype())
	}

	out, err := G.HadamardDiv(sum, n)
	if err != nil {
		return nil, fmt.Errorf("reduceMean: could not divide by number of "+
			"elements: %v", err)
	}

	return out, err
}

// ReduceSub calculates the difference along axis and squeezes all
// axes
func ReduceSub(x *G.Node, axis int) (*G.Node, error) {
	return ReduceAlong(x, axis, G.Sub)
}

// ReduceAlong iteratively applies f to the first two rows of x along
// axis, replacing the first row of x along axis by the output of this
// operation at each iteration.
func ReduceAlong(x *G.Node, axis int, f func(*G.Node, *G.Node) (*G.Node, error)) (*G.Node, error) {
	if axis >= len(x.Shape()) {
		return nil, fmt.Errorf("reduceAlong: axis out of range [%v] with "+
			"length %v", axis, len(x.Shape()))
	}

	// Squeeze out all dimensions of length 1
	var err error
	x, err = SqueezeAllBut(x, axis)
	if err != nil {
		return nil, fmt.Errorf("reduceAlong: could not squeeze dimensions: %v",
			err)
	}
	length := x.Shape()[axis]

	// If the length is 1, then just reshape
	if length == 1 {
		return Squeeze(x, axis)
	}

	// Get the first row along the axis
	ind := make([]tensor.Slice, len(x.Shape()))
	ind[axis] = G.S(0)
	row, err := G.Slice(x, ind...)
	if err != nil {
		return nil, fmt.Errorf("reduceAlong: axis does not have any elements")
	}

	// Add each consecutive row
	for i := 1; i < length; i++ {
		ind[axis] = G.S(i)
		nextRow, err := G.Slice(x, ind...)
		if err != nil {
			return nil, fmt.Errorf("reduceAlong: could not get row %v: %v",
				i, err)
		}

		row, err = f(row, nextRow)
		if err != nil {
			return nil, fmt.Errorf("reduceAlong: could not compute f "+
				"along rows: %v", err)
		}
	}

	fmt.Println("done", row.Shape())
	return row, nil
}

// ReduceAdd calculates the sum along axis and squeezes the axis out
func ReduceAdd(x *G.Node, axis int) (*G.Node, error) {
	return ReduceAlong(x, axis, G.Add)
	// fmt.Println("Input shape:", x.Shape())
	// if axis >= len(x.Shape()) {
	// 	return nil, fmt.Errorf("reduceAdd: axis out of range [%v] with "+
	// 		"length %v", axis, len(x.Shape()))
	// }

	// // Squeeze out all dimensions of length 1
	// var err error
	// x, err = SqueezeAllBut(x, axis)
	// if err != nil {
	// 	return nil, fmt.Errorf("reduceAdd: could not squeeze dimensions: %v",
	// 		err)
	// }

	// fmt.Println("OUT=====", x.Shape())

	// fmt.Println("Adjusted input shape:", x.Shape())
	// fmt.Println("Axis:", axis)
	// length := x.Shape()[axis]
	// fmt.Println("Length:", length)

	// // If the length is 1, then just reshape
	// if length == 1 {
	// 	fmt.Println("LENGTH 1")
	// 	return Squeeze(x, axis)
	// }

	// // Get the first row along the axis
	// ind := make([]tensor.Slice, len(x.Shape()))
	// ind[axis] = G.S(0)
	// row, err := G.Slice(x, ind...)
	// if err != nil {
	// 	return nil, fmt.Errorf("reduceAdd: axis does not have any elements")
	// }

	// // Add each consecutive row
	// for i := 1; i < length; i++ {
	// 	ind[axis] = G.S(i)
	// 	nextRow, err := G.Slice(x, ind...)
	// 	fmt.Println("Next row shape:", nextRow.Shape())
	// 	if err != nil {
	// 		return nil, fmt.Errorf("reduceAdd: could not get row %v: %v",
	// 			i, err)
	// 	}

	// 	row, err = G.Add(row, nextRow)
	// 	if err != nil {
	// 		return nil, fmt.Errorf("reduceAdd: could not add rows: %v",
	// 			err)
	// 	}
	// }

	// fmt.Println("done", row.Shape())
	// return row, nil
}

// ReduceDiv calculates the quotient along axis and squeezes the axis
// out
func ReduceDiv(x *G.Node, axis int) (*G.Node, error) {
	return ReduceAlong(x, axis, G.HadamardDiv)
}

// ReduceProd calculates the product along axis and squeezes the axis
// out
func ReduceProd(x *G.Node, axis int) (*G.Node, error) {
	return ReduceAlong(x, axis, G.HadamardProd)
}

// Repeat repeats the elements of x along axis repeats times. This
// function is conceptually similar to Numpy's repeat function and
// PyTorch's repeat_interleave function.
func Repeat(x *G.Node, axis, repeats int) (*G.Node, error) {
	if x.Shape().Dims() == 0 {
		return nil, fmt.Errorf("repeat: cannot repeat non-tensor node")
	}
	if axis >= x.Shape().Dims() {
		return nil, fmt.Errorf("repeat: cannot have axis (%v) > dims (%v)",
			axis, x.Shape().Dims())
	}

	op, err := newRepeatOp(axis, x.Shape().Dims(), repeats)
	if err != nil {
		return nil, fmt.Errorf("repeat: %v", err)
	}

	return G.ApplyOp(op, x)
}

// Clamp clamps a node's values to be between min and max. This function
// can clamp a tensor storing float64's, float32's, or any integer
// type, but is only differentiable if the tensor stores floating point
// types. If clamping a tensor of an integer type, the returned tensor
// will have type tensor.Int, regardless of the input tensor integer
// type. If passGradient is true, then the gradient is passed through
// the clamping operation:
//
//				⎧ 1 if min <= x <= max
//		grad =  ⎨
//				⎩ 1 otherwise
//
// Otherwise, the regular clamp gradient is used:
//
//				⎧ 1 if min <= x <= max
//		grad =  ⎨
//				⎩ 0 otherwise
//
func Clamp(x *G.Node, min, max interface{}, passGradient bool) (*G.Node,
	error) {
	op, err := newClampOp(min, max, passGradient)
	if err != nil {
		return nil, fmt.Errorf("clamp: %v", err)
	}

	return G.ApplyOp(op, x)
}

// Argsort returns the indices that would sort x along axis
func Argsort(x *G.Node, axis int) (*G.Node, error) {
	op := newArgsortOp(axis, x.Shape().Dims())

	return G.ApplyOp(op, x)
}

// Erfinv computes the element-wise inverse error function
func Erfinv(x *G.Node) (*G.Node, error) {
	op := newErfinvOp()

	return G.ApplyOp(op, x)
}

// Erf computes the element-wise error function
func Erf(x *G.Node) (*G.Node, error) {
	op := newErfOp()

	return G.ApplyOp(op, x)
}

// Erfc computes the element-wise complementary error function
func Erfc(x *G.Node) (*G.Node, error) {
	op := newErfOp()

	retVal, err := G.ApplyOp(op, x)
	if err != nil {
		return nil, fmt.Errorf("erfc: %v", err)
	}

	var one *G.Node
	switch x.Dtype() {
	case G.Float64:
		one = G.NewScalar(
			x.Graph(),
			G.Float64,
			G.WithValue(1.0),
		)

	case G.Float32:
		one = G.NewScalar(
			x.Graph(),
			G.Float32,
			G.WithValue(float32(1.0)),
		)
	}

	return G.Sub(one, retVal)
}

// Clip performs an element-wise clipping of all values in a node
// to be within [max, min]. This is similar to the Clamp operation,
// but is implemented differently. The Clamp operation should be
// used in place of this one whenever possible.
func Clip(value *G.Node, min, max float64) (retVal *G.Node, err error) {
	// Construct clipping nodes
	var minNode, maxNode *G.Node
	switch value.Dtype() {
	case G.Float32:
		minNode = G.NewScalar(
			value.Graph(),
			G.Float32,
			G.WithValue(float32(min)),
			G.WithName("clip_min"),
		)
		maxNode = G.NewScalar(
			value.Graph(),
			G.Float32,
			G.WithValue(float32(max)),
			G.WithName("clip_max"),
		)
	case G.Float64:
		minNode = G.NewScalar(
			value.Graph(),
			G.Float64,
			G.WithValue(min),
			G.WithName("clip_min"),
		)
		maxNode = G.NewScalar(
			value.Graph(),
			G.Float64,
			G.WithValue(max),
			G.WithName("clip_max"),
		)
	}

	// Check if its the min value
	minMask, err := G.Lt(value, minNode, true)
	if err != nil {
		return nil, err
	}
	minVal, err := G.HadamardProd(minNode, minMask)
	if err != nil {
		return nil, err
	}

	// Check if its the given value
	isMaskGt, err := G.Gt(value, minNode, true)
	if err != nil {
		return nil, err
	}
	isMaskLt, err := G.Lt(value, maxNode, true)
	if err != nil {
		return nil, err
	}
	isMask, err := G.HadamardProd(isMaskGt, isMaskLt)
	if err != nil {
		return nil, err
	}
	isVal, err := G.HadamardProd(value, isMask)
	if err != nil {
		return nil, err
	}

	// Check if its the max value
	maxMask, err := G.Gt(value, maxNode, true)
	if err != nil {
		return nil, err
	}
	maxVal, err := G.HadamardProd(maxNode, maxMask)
	if err != nil {
		return nil, err
	}
	return G.ReduceAdd(G.Nodes{minVal, isVal, maxVal})
}

// Min returns the min value between the nodes. If values are equal
// the first value is returned
func Min(a *G.Node, b *G.Node) (retVal *G.Node, err error) {
	aMask, err := G.Lte(a, b, true)
	if err != nil {
		return nil, err
	}
	aVal, err := G.HadamardProd(a, aMask)
	if err != nil {
		return nil, err
	}

	bMask, err := G.Lt(b, a, true)
	if err != nil {
		return nil, err
	}
	bVal, err := G.HadamardProd(b, bMask)
	if err != nil {
		return nil, err
	}
	return G.Add(aVal, bVal)
}

// Max value between the nodes. If values are equal the first value
// is returned.
func Max(a *G.Node, b *G.Node) (retVal *G.Node, err error) {
	aMask, err := G.Gte(a, b, true)
	if err != nil {
		return nil, err
	}
	aVal, err := G.HadamardProd(a, aMask)
	if err != nil {
		return nil, err
	}

	bMask, err := G.Gt(b, a, true)
	if err != nil {
		return nil, err
	}
	bVal, err := G.HadamardProd(b, bMask)
	if err != nil {
		return nil, err
	}
	return G.Add(aVal, bVal)
}

// AddFauxF32 adds a the faux zero value 1e-6.
func AddFauxF32(n *G.Node) (retVal *G.Node, err error) {
	faux := G.NewScalar(n.Graph(), G.Float32, G.WithValue(float32(1e-6)))
	return G.BroadcastAdd(faux, n, []byte{}, []byte{})
}

// LogSumExp calculates the log of the summation of exponentials of
// all logits along the given axis.
//
// Use this in place of Gorgonia's LogSumExp, which has the final sum
// and log interchanged, which is incorrect.
func LogSumExp(logits *G.Node, along int) *G.Node {
	max := G.Must(G.Max(logits, along))

	exponent := G.Must(G.BroadcastSub(logits, max, nil, []byte{1}))
	exponent = G.Must(G.Exp(exponent))

	sum := G.Must(G.Sum(exponent, along))
	log := G.Must(G.Log(sum))

	return G.Must(G.Add(max, log))
}

// Prod calculates the product of a Node along an axis
func Prod(input *G.Node, along int) *G.Node {
	shape := input.Shape()

	// Calculate the first columns along the axis along
	dims := make([]tensor.Slice, len(shape))
	for i := 0; i < len(shape); i++ {
		if i == along {
			dims[i] = G.S(0, 1, 1)
		}
	}
	prod := G.Must(G.Slice(input, dims...))

	for i := 1; i < input.Shape()[along]; i++ {
		// Calculate the column that should be multiplied next
		for j := 0; j < len(shape); j++ {
			if j == along {
				dims[j] = G.S(i)
			}
		}

		s := G.Must(G.Slice(input, dims...))
		prod = G.Must(G.HadamardProd(prod, s))
	}
	return prod
}
