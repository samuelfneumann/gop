package distribution

import (
	"fmt"

	"github.com/samuelfneumann/gop"
	G "gorgonia.org/gorgonia"
)

// Event dims always taken from right...

type IID struct {
	Distribution
	dims int // The number of batch dimensions to interpret as events
}

func NewIID(d Distribution, dims int) *IID {
	return &IID{d, dims}
}

// SetDims sets the number of event dims
func (i *IID) SetDims(dims int) {
	i.dims = dims
}

func (i *IID) Prob(x *G.Node) (*G.Node, error) {
	if x.Dims() < i.dims {
		return nil, fmt.Errorf("prob: expected dims >= %v but got %v", i.dims,
			x.Dims())
	}

	x, err := i.Distribution.Prob(x)
	if err != nil {
		return nil, fmt.Errorf("prob: could not compute iid prob: %v", err)
	}

	// Combine event dims
	for j := 0; j < i.dims; j++ {
		x, err = gop.ReduceProd(x, x.Dims()-1, true)
		if err != nil {
			return nil, fmt.Errorf("prob: could not combine event dims: %v",
				err)
		}
	}

	return x, nil
}

func (i *IID) LogProb(x *G.Node) (*G.Node, error) {
	if x.Dims() < i.dims {
		return nil, fmt.Errorf("logProb: expected dims >= %v but got %v", i.dims,
			x.Dims())
	}

	x, err := i.Distribution.LogProb(x)
	if err != nil {
		return nil, fmt.Errorf("logProb: could not compute iid prob: %v", err)
	}

	// Combine event dims
	for j := 0; j < i.dims; j++ {
		x, err = gop.ReduceAdd(x, x.Dims()-1, true)
		if err != nil {
			return nil, fmt.Errorf("logProb: could not combine event dims: %v",
				err)
		}
	}

	return x, nil
}

func (i *IID) Entropy() (*G.Node, error) {
	x, err := i.Distribution.Entropy()
	if err != nil {
		return nil, fmt.Errorf("entropy: could not take entropy of each "+
			"i.i.d. variable: %v", err)
	}

	// Combine event dims
	for j := 0; j < i.dims; j++ {
		x, err = gop.ReduceAdd(x, x.Dims()-1, true)
		if err != nil {
			return nil, fmt.Errorf("entropy: could not combine event dims: %v",
				err)
		}
	}

	return x, nil
}

func (i *IID) Cdf(x *G.Node) (*G.Node, error) {
	if x.Dims() < i.dims {
		return nil, fmt.Errorf("cdf: expected dims >= %v but got %v", i.dims,
			x.Dims())
	}

	x, err := i.Distribution.Cdf(x)
	if err != nil {
		return nil, fmt.Errorf("cdf: could not compute iid cdf: %v", err)
	}

	// Combine event dims
	for j := 0; j < i.dims; j++ {
		x, err = gop.ReduceProd(x, x.Dims()-1, true)
		if err != nil {
			return nil, fmt.Errorf("cdf: could not combine event dims: %v",
				err)
		}
	}

	return x, nil
}
