package distribution

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

func NormalRand(mean, stddev *G.Node, seed uint64,
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
