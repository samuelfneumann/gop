// Package distribution provides probability distributions
package distribution

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Quantiler is a Distribution that can return the inverse of the CDF
// function, sometimes called the quantile function.
type Quantiler interface {
	Distribution
	Quantile(*G.Node) (*G.Node, error)
}

// Distribution is a probability distribution
type Distribution interface {
	// Cdf returns the cumulative probability density of mass
	// of the node. The shape of the node must be
	// compatible with the shape of the distribution.
	//
	// If the node has one more dimension than the dimensions
	// of the distribution, then the first dimension
	// of the input node is taken to be the batch dimension.
	// Otherwise, the node must have the same number of
	// dimensions as samples generated from the
	// distribution
	Cdf(*G.Node) (*G.Node, error)

	Entropy() (*G.Node, error)
	Shape() tensor.Shape

	// LogProb returns the log of the probability density of
	// mass of the node. The shape of the node must be
	// compatible with the shape of the distribution.
	//
	// If the node has one more dimension than the dimensions
	// of the distribution, then the first dimension
	// of the input node is taken to be the batch dimension.
	// Otherwise, the node must have the same number of
	// dimensions as samples generated from the
	// distribution
	LogProb(*G.Node) (*G.Node, error)

	// Prob returns the probability density or mass of the
	// node. The shape of the node must be compatible with
	// the shape of the distribution.
	//
	// If the node has one more dimension than the dimensions
	// of the distribution, then the first dimension
	// of the input node is taken to be the batch dimension.
	// Otherwise, the node must have the same number of
	// dimensions as samples generated from the
	// distribution
	Prob(*G.Node) (*G.Node, error)

	Mean() *G.Node
	StdDev() *G.Node
	Variance() *G.Node

	// Sample returns a node that generates samples
	// from some distribution each time the node is passed. This
	// function is not differentiable.
	Sample(samples int) (*G.Node, error)

	// Rsample returns a node that generates reparameterized samples
	// from some distribution each time the node is passed. This
	// function is differentiable.
	Rsample(samples int) (*G.Node, error)

	// Returns whether the distribution has reparameterized samples or
	// not
	HasRsample() bool
}
