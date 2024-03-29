package distribution

import (
	"fmt"
	"hash"

	"golang.org/x/exp/rand"

	"github.com/chewxy/hm"
	"github.com/samuelfneumann/gop"
	"gonum.org/v1/gonum/stat/distuv"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// normalSampleOp is an operation that samples from a normal
// distribution whenever the node is passed through. The normalSampleOp
// is not differentiable.
type normalSampleOp struct {
	dt         tensor.Dtype
	shape      tensor.Shape
	dist       distuv.Normal
	source     rand.Source
	numSamples int
}

// newNormalSampleOp returns a new normalSampleOp
func newNormalSampleOp(dt tensor.Dtype, seed uint64, numSamples int,
	shape ...int) (*normalSampleOp, error) {
	if dt != tensor.Float64 && dt != tensor.Float32 {
		return nil, fmt.Errorf("newGaussianSampleOp: dtype %v not supported",
			dt)
	}

	if numSamples < 1 {
		return nil, fmt.Errorf("cannot samples %v < 1 samples", numSamples)
	}

	source := rand.NewSource(seed)

	return &normalSampleOp{
		dt:     dt,
		shape:  tensor.Shape(shape),
		source: source,
		dist: distuv.Normal{
			Mu:    0.0,
			Sigma: 1.0,
			Src:   source,
		},
		numSamples: numSamples,
	}, nil
}

// Arity implements the gorgonia.Op interface
func (n *normalSampleOp) Arity() int { return 2 }

// Type implements the gorgonia.Op interface
func (n *normalSampleOp) Type() hm.Type {
	in := G.TensorType{
		Dims: n.shape.Dims(),
		Of:   n.dt,
	}
	out := G.TensorType{
		Dims: n.shape.Dims() + 1,
		Of:   n.dt,
	}

	return hm.NewFnType(in, in, out)
}

// InferShape implements the gorgonia.Op interface
func (n *normalSampleOp) InferShape(...G.DimSizer) (tensor.Shape, error) {
	return append([]int{n.numSamples}, n.shape...), nil
}

// ReturnsPtr implements the gorgonia.Op interface
func (n *normalSampleOp) ReturnsPtr() bool { return false }

// CallsExtern implements the gorgonia.Op interface
func (n *normalSampleOp) CallsExtern() bool { return false }

// OverwritesInput implements the gorgonia.Op interface
func (n *normalSampleOp) OverwritesInput() int { return -1 }

// String implements the fmt.Stringer interface
func (n *normalSampleOp) String() string {
	return fmt.Sprintf("NormalRand{shape=%v}()", append([]int{n.numSamples},
		n.shape...))
}

// WriteHash implements the gorgonia.Op interface
func (n *normalSampleOp) WriteHash(h hash.Hash) {
	fmt.Fprint(h, n.String())
}

// Hashcode implements the gorgonia.Op interface
func (n *normalSampleOp) Hashcode() uint32 {
	return gop.SimpleHash(n)
}

// Do implements the gorgonia.Op interface
func (n *normalSampleOp) Do(inputs ...G.Value) (G.Value, error) {
	if err := n.checkInputs(inputs...); err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	out := tensor.NewDense(
		n.dt,
		append([]int{n.numSamples}, n.shape...),
	)

	mean := inputs[0].(tensor.Tensor)
	std := inputs[1].(tensor.Tensor)

	// Create the distributions and sample
	for i := 0; i < mean.Size(); i++ {
		coords, err := tensor.Itol(i, mean.Shape(), mean.Strides())
		if err != nil {
			return nil, fmt.Errorf("do: could not get coords at index %v", i)
		}

		currentMean, err := mean.At(coords...)
		if err != nil {
			return nil, fmt.Errorf("do: could not get mean at index %v", i)
		}
		currentStd, err := std.At(coords...)
		if err != nil {
			return nil, fmt.Errorf("do: could not get std at index %v", i)
		}

		n.dist.Mu = currentMean.(float64)
		n.dist.Sigma = currentStd.(float64)

		outCoords := append([]int{0}, coords...)
		for j := 0; j < n.numSamples; j++ {
			outCoords[0] = j

			if n.dt == tensor.Float64 {
				out.SetAt(n.dist.Rand(), outCoords...)
			} else {
				out.SetAt(float32(n.dist.Rand()), outCoords...)
			}
		}
	}

	return out, nil
}

// checkInputs returns an error if inputs is an illegal input for the
// receiver
func (n *normalSampleOp) checkInputs(inputs ...G.Value) error {
	if err := gop.CheckArity(n, len(inputs)); err != nil {
		return err
	}

	mean := inputs[0].(tensor.Tensor)
	if mean == nil {
		return fmt.Errorf("cannot sample from nil mean")
	} else if mean.Size() == 0 {
		return fmt.Errorf("cannot sample from empty mean tensor")
	} else if !mean.Shape().Eq(n.shape) {
		return fmt.Errorf("expected mean to have shape %v but got %v",
			n.shape, mean.Shape())
	} else if !mean.Dtype().Eq(n.dt) {
		return fmt.Errorf("expected mean to have dtype %v but got %v",
			n.dt, mean.Dtype())
	}

	stddev := inputs[1].(tensor.Tensor)
	if stddev == nil {
		return fmt.Errorf("cannot sample from nil stddev")
	} else if stddev.Size() == 0 {
		return fmt.Errorf("cannot sample from empty stddev tensor")
	} else if !stddev.Shape().Eq(n.shape) {
		return fmt.Errorf("expected stddev to have shape %v but got %v",
			n.shape, stddev.Shape())
	} else if !stddev.Dtype().Eq(n.dt) {
		return fmt.Errorf("expected stddev to have dtype %v but got %v",
			n.dt, stddev.Dtype())
	}

	return nil
}
