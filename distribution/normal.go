package distribution

import (
	"fmt"
	"math"

	"github.com/samuelfneumann/gop"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TODO: make work with float32

// Normal is a univariate normal distribution, which may hold
// a batch of normal distributions simultaneously. If a Normal is
// created with a tensor mean and tensor standard deviation, then
// each elemnt of the mean and standard deviation vectors defines a
// different distribution element-wise. For example, consider if we
// use a 1-tensor for the mean and standard deviation:
//
//		mean   := [m_1, m_2, ..., m_N]
//		stddev := [s_1, s_2, ..., s_N]
//
// Then the Normal is considered to hold the following distributions:
//
//		[𝒩(m_1, s_1), 𝒩(m_2, s_2), ..., 𝒩(m_N, s_N)]
//
// If the mean and standard deviation are scalars or vectors of 1
// element, then a single Normal distribution is used.
//
// The shape of the mean and standard deviation tensors consitutie
// the shape of the Normal. E.g. if the mean has shape (3, 2, 5), then
// so does the Normal.
//
// Any input to any method of the Normal must have a shape that is
// consistent with the shape of the Normal. That is, the input must
// have the exact same shape as the Normal, except for possibly the
// batch dimension, which is dimension 0 always. If a batch dimension
// is present, then the method will be run on each sample in the batch.
// Given a Normal with shape (n_1, n_2, ..., n_M), the following are
// legal shapes for an input:
//
// 1. (n_1, n_2, ..., n_M)
// 2. (a, n_1, n_2, ..., n_M) for ∀a ∈ ℕ-{0}
//
// Normal supports the following data types:
// - tensor.Float64
type Normal struct {
	mean    *G.Node
	meanVal G.Value

	stddev    *G.Node
	stddevVal G.Value

	// zeroMean   *G.Node
	// unitStddev *G.Node
	// stdNormal  *G.Node

	seed uint64
}

// NewNormal returns a new Normal.
func NewNormal(mean, stddev *G.Node, seed uint64) (*Normal, error) {
	if !mean.Shape().Eq(stddev.Shape()) {
		return nil, fmt.Errorf("newNormal: expected mean and stddev to "+
			"have the same shape but got %v and %v", mean.Shape(),
			stddev.Shape())
	}

	if mean.Dtype() != stddev.Dtype() {
		return nil, fmt.Errorf("newNormal: expected mean and stddev to "+
			"have the same data type but got %v and %v", mean.Dtype(),
			stddev.Dtype())
	} else if mean.Dtype() != tensor.Float64 {
		return nil, fmt.Errorf("newNormal: data type %v unsupported",
			mean.Dtype())
	}

	var err error
	if mean.IsScalar() {
		mean, err = G.Reshape(mean, []int{1})
		if err != nil {
			return nil, fmt.Errorf("newNormal: could not expand mean to "+
				"shape (1): %v", err)
		}
		stddev, err = G.Reshape(stddev, []int{1})
		if err != nil {
			return nil, fmt.Errorf("newNormal: could not expand stddev to "+
				"shape (1): %v", err)
		}
	}

	normal := &Normal{
		mean:   mean,
		stddev: stddev,
		seed:   seed,
	}

	G.Read(normal.mean, &normal.meanVal)
	G.Read(normal.stddev, &normal.stddevVal)

	return normal, nil
}

// Prob calculates the probability density of x.
//
// If the receiver's mean and standard deviation nodes are scalars, then
// if x is a vector, this function treats it as a batch of values to
// compute the probability density of. In this case, the density
// will be calculated element-wise for each value in the vector x with
// the same mean and standard deviation.
//
// If the mean and standard deviation of the receiver are tensors,
// then the receiver is assumed to hold N normal distributions,
// where N is the number of elements in the mean or standard
// deviation vectors respectively. In this case, an input tensor x
// should have the same shape as the mean and standard
// deviation tensors, except for perhaps the batch dimension (dim 0).
// If not, an error is returned.
// For example, if the mean and stddev of the Normal are vectors:
//
//		mean   := [m_1, m_2, ..., m_N]
//		stddev := [s_1, s_2, ..., s_N]
//
// Then x should be of the form:
//		x 	   := ⎡x_11, x_21, ..., x_N1⎤ ⎫
//				  ⎢x_12, x_22, ..., x_N2⎥ ⎥
//				  ⎢... ... ... ..., ... ⎥ ⎬ ← Batch Dimension
//				  ⎢... ... ... ..., ... ⎥ ⎥
//				  ⎣x_1M, x_2M, ... x_NM ⎦ ⎭
//
// In such a case, there are M samples considered to be in a batch, and
// there are N separate univariate normal distributions.
func (n *Normal) Prob(x *G.Node) (*G.Node, error) {
	x, err := n.fixShape(x)
	if err != nil {
		return nil, fmt.Errorf("prob: %v", err)
	}

	two := x.Graph().Constant(G.NewF64(2.0))
	negativeHalf := x.Graph().Constant(G.NewF64(-0.5))
	rootTwoPi := x.Graph().Constant(G.NewF64(math.Sqrt(math.Pi * 2.)))

	if n.isBatch(x) {
		// Calculate probability of batch
		batchDim := []byte{0}
		x = G.Must(G.BroadcastSub(x, n.mean, nil, batchDim))
		x = G.Must(G.BroadcastHadamardDiv(x, n.stddev, nil, batchDim))
		x = G.Must(G.Pow(x, two))
		x = G.Must(G.HadamardProd(negativeHalf, x))
		x = G.Must(G.Exp(x))
		x = G.Must(G.BroadcastHadamardDiv(x, n.stddev, nil, batchDim))
		x = G.Must(G.HadamardDiv(x, rootTwoPi))
	} else {
		// Calculate probability of single sample
		x = G.Must(G.Sub(x, n.mean))
		x = G.Must(G.HadamardDiv(x, n.stddev))
		x = G.Must(G.Pow(x, two))
		x = G.Must(G.HadamardProd(negativeHalf, x))
		x = G.Must(G.Exp(x))
		x = G.Must(G.HadamardDiv(x, n.stddev))
		x = G.Must(G.HadamardDiv(x, rootTwoPi))
	}

	return x, nil
}

// LogProb calculates the log probability of x. The shape of x is
// treated in the same way as the Prob() method.
func (n *Normal) LogProb(x *G.Node) (*G.Node, error) {
	x, err := n.fixShape(x)
	if err != nil {
		return nil, fmt.Errorf("logProb: %v", err)
	}

	two := x.Graph().Constant(G.NewF64(2.0))
	negativeHalf := x.Graph().Constant(G.NewF64(-0.5))
	lnRootTwoPi := x.Graph().Constant(G.NewF64(math.Log(math.Sqrt(
		math.Pi * 2.))))

	if n.isBatch(x) {
		// Calculate probability of batch
		batchDim := []byte{0}
		x = G.Must(G.BroadcastSub(x, n.mean, nil, batchDim))
		x = G.Must(G.BroadcastHadamardDiv(x, n.stddev, nil, batchDim))
		x = G.Must(G.Pow(x, two))
		x = G.Must(G.HadamardProd(negativeHalf, x))
		lnStd := G.Must(G.Log(n.stddev))
		x = G.Must(G.BroadcastSub(x, lnStd, nil, batchDim))
		x = G.Must(G.Sub(x, lnRootTwoPi))
	} else {
		// Calculate probability of single sample
		x = G.Must(G.Sub(x, n.mean))
		x = G.Must(G.HadamardDiv(x, n.stddev))
		x = G.Must(G.Pow(x, two))
		x = G.Must(G.HadamardProd(negativeHalf, x))
		lnStd := G.Must(G.Log(n.stddev))
		x = G.Must(G.Sub(x, lnStd))
		x = G.Must(G.Sub(x, lnRootTwoPi))
	}

	return x, nil
}

// Cdf computes the cumulative distribution function of x. The shape
// of x is treated in the same way as the Prob() method.
func (n *Normal) Cdf(x *G.Node) (*G.Node, error) {
	x, err := n.fixShape(x)
	if err != nil {
		return nil, fmt.Errorf("prob: %v", err)
	}

	if x.IsScalar() {
		x, err = G.Reshape(x, []int{1})
		if err != nil {
			return nil, fmt.Errorf("prob: could not reshape x: %v", err)
		}
	}

	rootTwo := x.Graph().Constant(G.NewF64(math.Sqrt(2.0)))
	one := x.Graph().Constant(G.NewF64(1.0))
	half := x.Graph().Constant(G.NewF64(0.5))

	if n.isBatch(x) {
		// Calculate probability of batch
		batchDim := []byte{0}
		x = G.Must(G.BroadcastSub(x, n.mean, nil, batchDim))
		x = G.Must(G.HadamardDiv(x, rootTwo))
		x = G.Must(G.BroadcastHadamardDiv(x, n.stddev, nil, batchDim))
		x = G.Must(gop.Erf(x))
		x = G.Must(G.Add(one, x))
		x = G.Must(G.HadamardProd(half, x))
	} else {
		// Calculate the probability density of a single observation
		x = G.Must(G.Sub(x, n.mean))
		x = G.Must(G.HadamardDiv(x, rootTwo))
		x = G.Must(G.HadamardDiv(x, n.stddev))
		x = G.Must(gop.Erf(x))
		x = G.Must(G.Add(one, x))
		x = G.Must(G.HadamardProd(half, x))
	}

	return x, nil
}

// Cdfinv computes the inverse cumulative distribution function at
// probability p. The shape of p is treated in the same way as the
// Prob() method.
func (n *Normal) Cdfinv(p *G.Node) (*G.Node, error) {
	p, err := n.fixShape(p)
	if err != nil {
		return nil, fmt.Errorf("prob: %v", err)
	}

	if p.IsScalar() {
		p, err = G.Reshape(p, []int{1})
		if err != nil {
			return nil, fmt.Errorf("prob: could not reshape x: %v", err)
		}
	}

	rootTwo := p.Graph().Constant(G.NewF64(math.Sqrt(2.0)))
	one := p.Graph().Constant(G.NewF64(1.0))
	two := p.Graph().Constant(G.NewF64(2.0))

	if n.isBatch(p) {
		// Calculate probability of batch
		batchDim := []byte{0}
		p = G.Must(G.HadamardProd(two, p))
		p = G.Must(G.Sub(p, one))
		p = G.Must(gop.Erfinv(p))
		p = G.Must(G.HadamardProd(p, rootTwo))
		p = G.Must(G.BroadcastHadamardProd(p, n.stddev, nil, batchDim))
		p = G.Must(G.BroadcastAdd(p, n.mean, nil, batchDim))
	} else {
		// Calculate the probability density of a single observation
		p = G.Must(G.HadamardProd(two, p))
		p = G.Must(G.Sub(p, one))
		p = G.Must(gop.Erfinv(p))
		p = G.Must(G.HadamardProd(p, rootTwo))
		p = G.Must(G.HadamardProd(p, n.stddev))
		p = G.Must(G.Add(n.mean, p))
	}

	return p, nil
}

// Shape returns the number of distributions stored by the receiver
func (n *Normal) Shape() tensor.Shape {
	return n.mean.Shape()
}

// Variance returns the variance of the distribution(s) stored by the
// receiver
func (n *Normal) Variance() *G.Node {
	two := n.mean.Graph().Constant(G.NewF64(2.0))
	return G.Must(G.Pow(n.stddev, two))
}

// StdDev returns the standard deviation of the distribution(s)
// stored by the receiver
func (n *Normal) StdDev() *G.Node {
	return n.stddev
}

// Mean returns the mean of the distribution(s) stored by the
// receiver
func (n *Normal) Mean() *G.Node {
	return n.mean
}

// Entropy returns the entropy of the distribution(s) stored by the
// receiver
func (n *Normal) Entropy() *G.Node {
	half := n.mean.Graph().Constant(G.NewF64(0.5))
	twoPi := n.mean.Graph().Constant(G.NewF64(math.Pi * 2.0))
	two := n.mean.Graph().Constant(G.NewF64(2.0))

	entropy := G.Must(G.Pow(n.stddev, two))
	entropy = G.Must(G.HadamardProd(entropy, twoPi))
	entropy = G.Must(G.Log(entropy))
	entropy = G.Must(G.HadamardProd(half, entropy))
	entropy = G.Must(G.Add(entropy, half))

	return entropy
}

func (n *Normal) HasRsample() bool { return true }

// // TODO: Rsample uses batch dim as dim 0, make all other functions
// // TODO: use this as well
// func (n *Normal) Rsample(samples int) (*G.Node, error) {
// 	if n.zeroMean == nil || n.unitStddev == nil {
// 		// Lazy instantiation of zero mean and unit variance
// 		size := tensor.ProdInts(n.mean.Shape())

// 		zeroMean := tensor.NewDense(
// 			tensor.Float64,
// 			n.mean.Shape(),
// 			tensor.WithBacking(make([]float64, size)),
// 		)
// 		n.zeroMean = G.NewTensor(
// 			n.mean.Graph(),
// 			zeroMean.Dtype(),
// 			zeroMean.Dims(),
// 			G.WithValue(zeroMean),
// 			G.WithName("zeroMean"),
// 		)

// 		unitStddev := tensor.NewDense(
// 			tensor.Float64,
// 			n.stddev.Shape(),
// 			tensor.WithBacking(ones(size)),
// 		)
// 		n.unitStddev = G.NewTensor(
// 			n.stddev.Graph(),
// 			unitStddev.Dtype(),
// 			unitStddev.Dims(),
// 			G.WithValue(unitStddev),
// 			G.WithName("unitStddev"),
// 		)
// 		n.stdNormal = G.Must(NormalRand(n.mean, n.stddev, n.seed, samples))
// 		fmt.Println("SHAPES:", n.stdNormal.Shape(), n.stddev.Shape())
// 	}

// 	// Reparameterization trick
// 	// var out *G.Node
// 	// if samples > 1 {
// 	// 	out = G.Must(G.BroadcastHadamardProd(n.stdNormal, n.stddev, nil,
// 	// 		[]byte{0}))
// 	// 	out = G.Must(G.BroadcastAdd(out, n.mean, nil, []byte{0}))
// 	// } else {
// 	// 	out = G.Must(G.HadamardProd(n.stdNormal, n.stddev))
// 	// 	out = G.Must(G.Add(out, n.mean))
// 	// }

// 	// return out, nil
// 	return nil, nil
// }

func (n *Normal) Sample(samples int) (*G.Node, error) {
	return NormalRand(n.mean, n.stddev, n.seed, samples)
}

// isBatch returns whether x is a batch of samples to calculate some
// method on
func (n *Normal) isBatch(x *G.Node) bool {
	return !x.Shape().Eq(n.mean.Shape())
}

// fixShape adjusts the shape of x so that it can be used in some
// method. It returns an error indicating if x is of an invalid shape
// which could not be adjusted.
func (n *Normal) fixShape(x *G.Node) (*G.Node, error) {
	if x.IsScalar() && n.mean.Shape()[0] == 1 {
		return G.Reshape(x, []int{1})

	} else if len(x.Shape()) == 1 && n.mean.Shape()[0] == 1 {
		// When distribution shape was inputted as a scalar, then a
		// vector input x indicates a batch of samples -> reshape
		// so batch dims = 0 and shape of samples = dim 1
		return G.Reshape(x, []int{x.Shape()[0], 1})

	} else if n.isBatch(x) && !tensor.Shape(x.Shape()[1:]).Eq(n.Shape()) {
		msg := "expected shape to match distribution shape %v at all " +
			"dimensions except batch (dim 0) but got x shape %v"
		return nil, fmt.Errorf(msg, n.Shape(), x.Shape())

	} else if !n.isBatch(x) && !n.Shape().Eq(x.Shape()) {
		msg := "expected shape to match distribution shape %v but got %v"
		return nil, fmt.Errorf(msg, n.Shape(), x.Shape())
	}

	return x, nil
}
