package distribution

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// NormalSample returns numSamples samples from a normal distribution
// with mean mean and standard deviation stddev. The batch dimension
// is dimension 0 always.
//
// NormalSample is not a differentiable operation. For a differentiable
// sampling operation, see Normal.
func NormalSample(mean, stddev *G.Node, seed uint64,
	numSamples int) (*G.Node, error) {
	if mean.Dtype() != stddev.Dtype() {
		return nil, fmt.Errorf("normalRand: mean and stddev should have "+
			"same dtype but got %v and %v", mean.Dtype(), stddev.Dtype())
	}

	if !mean.Shape().Eq(stddev.Shape()) {
		return nil, fmt.Errorf("normalRand: mean and stddev should have "+
			"same shape but got %v and %v", mean.Shape(), stddev.Shape())
	}

	n, err := newNormalSampleOp(mean.Dtype(), seed, numSamples,
		mean.Shape()...)
	if err != nil {
		return nil, fmt.Errorf("normalRand: %v", err)
	}

	return G.ApplyOp(n, mean, stddev)
}
