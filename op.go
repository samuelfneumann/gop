// Package op provides extended operations for Gorgonia
package gop

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func Repeat(x *G.Node, axis, repeats int) (*G.Node, error) {
	op, err := newRepeatOp(axis, repeats)
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
//         { 1 if min <= x <= max
// grad =  {
//		   { 1 otherwise
//
// Otherwise, the regular clamp gradient is used:
//
//         { 1 if min <= x <= max
// grad =  {
//		   { 0 otherwise
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
	op := newArgsortOp(axis)

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
