// Package distribution provides probability distributions
package distribution

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Each of the following are SDOp's:
// CDF
// Entropy
// LogProb
// Prob
// InverseCDF
// RSample -- gotten automatically by reparam trick
// Mean() for some distributions, such as a Gaussian
// StdDev() -- for some distributions, such as a Gaussian
// Variance() -- for some distributions, such as a Gaussian

// Distribution is a probability distribution
type Distribution interface {
	Cdf(*G.Node) (*G.Node, error)
	ICdf(*G.Node) (*G.Node, error)

	Entropy() *G.Node
	EventShape() tensor.Shape

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

	// Sample generates a shape shaped sample or batch of
	// samples if the distribution is batched. This operation is
	// not differentiable.
	Sample(shape ...int) (*G.Node, error)

	// RSample generates a shape shaped sample or batch of samples
	// if the distribution is batched. This operation is differentiable.
	RSample(shape ...int) (*G.Node, error)

	// SampleN generates n samples or n batches of samples if the
	// distribution is batched. This operation is not differentiable.
	SampleN(n int) (*G.Node, error)
}
