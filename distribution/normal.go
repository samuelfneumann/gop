package distribution

import (
	"fmt"
	"math"

	"github.com/samuelfneumann/gop"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ! Alternative implementation would allow any shaped input and just
// ! always consider first dimension as the batch dimension for
// ! inputs. E.g. a mean/stddev of shape (3, 4, 12) then an input
// ! of shape (10, 3, 4, 12) would have 10 samples in the batch

// TODO: Make batch dimension be dimension 0 and allow input of any size
// this will be more consistent with the sample procedures which always use
// dimension 0 as the batch dimension. As well, the NormalRand function
// allows any input mean and stddev, but ALWAYS uses batch dim as dim 0.
// The Normal in this file should do this as well.

// TODO: make work with float32

// Normal is a univariate normal distribution, which may hold
// a batch of normal distributions simultaneously. If a Normal is
// created with a vector mean and vector standard deviation, then
// each elemnt of the mean and standard deviation vectors defines a
// different distribution. That is, if:
//
//		mean   := [m_1, m_2, ..., m_N]
//		stddev := [s_1, s_2, ..., s_N]
//
// Then the Normal is considered to hold the following distributions:
//
//		[𝒩(m_1, s_1), 𝒩(m_2, s_2), ..., 𝒩(m_N, s_N)]
//
// And each operation on the Normal should use an input with N
// dimensions.
//
// If the mean and standard deviation are scalars or vectors of 1
// element, then a single Normal distribution is used.
//
// The Normal can compute all its methods on batches of input data.
// If the Normal holds a single normal distribution (the mean and stddev
// are scalars or 1-vectors), then any vector input will be considered
// a batch upon which to compute the method. If the Normal holds N > 1
// normal distributions (the mean and stddev are N>1-vectors), then
// any matrix input will be considered a batch upon which to compute,
// with the batch dimension being the column dimension. The row
// dimension is considered as separate data for each of the N
// distributions, and the input matrix must have N rows exactly.
type Normal struct {
	mean    *G.Node
	meanVal G.Value

	stddev    *G.Node
	stddevVal G.Value

	zeroMean   *G.Node
	unitStddev *G.Node
	stdNormal  *G.Node

	seed uint64
}

// NewNormal returns a new Normal.
func NewNormal(mean, stddev *G.Node, seed uint64) (*Normal, error) {
	if !stddev.IsScalar() && !stddev.IsVector() {
		return nil, fmt.Errorf("newNormal: stddev must be a scalar or vector")
	}
	if !mean.IsScalar() && !mean.IsVector() {
		return nil, fmt.Errorf("newNormal: mean must be a scalar or vector")
	}
	if (mean.IsScalar() && !stddev.IsScalar()) || (!mean.IsScalar() &&
		stddev.IsScalar()) {
		return nil, fmt.Errorf("newNormal: mean and stddev must have the "+
			"same shape but got mean %v stddev %v", mean.Shape(),
			stddev.Shape())
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
// If the mean and standard deviation of the receiver are vectors,
// then the receiver is assumed to hold N normal distributions,
// where N is the number of elements in the mean or standard
// deviation vectors respectively. In this case, an input vector x
// should have the same size as the mean and standard
// deviation vectors. If not, an error is returned.
// If x is a matrix, then it should have
// the same number of rows as there are elements in the mean and
// standard deviation vectors. The number of columns of x is considered
// as the batch size of the samples to calculate the density of.
// That is if:
//
//		mean   := [m_1, m_2, ..., m_N]
//		stddev := [s_1, s_2, ..., s_N]
//
// Then x should be of the form:
//		x 	   := ⎡x_11, x_12, ..., x_1M⎤
//				  ⎢x_21, x_22, ..., x_2M⎥
//				  ⎢... ... ... ..., ... ⎥
//				  ⎣x_N1, x_N2, ... x_NM ⎦
//
// In such a case, there are M samples considered to be in a batch, and
// there are N separate univariate normal distributions.
func (n *Normal) Prob(x *G.Node) (*G.Node, error) {
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

	two := x.Graph().Constant(G.NewF64(2.0))
	negativeHalf := x.Graph().Constant(G.NewF64(-0.5))
	rootTwoPi := x.Graph().Constant(G.NewF64(math.Sqrt(math.Pi * 2.)))

	if n.isBatch(x) {
		// Calculate probability of batch
		var batchDim byte = 0
		if n.mean.Shape()[0] > 1 {
			batchDim = byte(1)
		}
		x = G.Must(G.BroadcastSub(x, n.mean, nil, []byte{batchDim}))
		x = G.Must(G.BroadcastHadamardDiv(x, n.stddev, nil, []byte{batchDim}))
		x = G.Must(G.Pow(x, two))
		x = G.Must(G.HadamardProd(negativeHalf, x))
		x = G.Must(G.Exp(x))
		x = G.Must(G.BroadcastHadamardDiv(x, n.stddev, nil, []byte{batchDim}))
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
		return nil, fmt.Errorf("prob: %v", err)
	}

	if x.IsScalar() {
		x, err = G.Reshape(x, []int{1})
		if err != nil {
			return nil, fmt.Errorf("prob: could not reshape x: %v", err)
		}
	}

	two := x.Graph().Constant(G.NewF64(2.0))
	negativeHalf := x.Graph().Constant(G.NewF64(-0.5))
	lnRootTwoPi := x.Graph().Constant(G.NewF64(math.Log(math.Sqrt(
		math.Pi * 2.))))

	if n.isBatch(x) {
		// Calculate probability of batch
		var batchDim []byte = []byte{0}
		if n.mean.Shape()[0] > 1 {
			batchDim = []byte{1}
		}
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
		var batchDim []byte = []byte{0}
		if n.mean.Shape()[0] > 1 {
			batchDim = []byte{1}
		}
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
		var batchDim []byte = []byte{0}
		if n.mean.Shape()[0] > 1 {
			batchDim = []byte{1}
		}
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

// TODO: Rsample uses batch dim as dim 0, make all other functions
// TODO: use this as well
func (n *Normal) Rsample(samples int) (*G.Node, error) {
	if n.zeroMean == nil || n.unitStddev == nil {
		// Lazy instantiation of zero mean and unit variance
		size := tensor.ProdInts(n.mean.Shape())

		zeroMean := tensor.NewDense(
			tensor.Float64,
			n.mean.Shape(),
			tensor.WithBacking(make([]float64, size)),
		)
		n.zeroMean = G.NewTensor(
			n.mean.Graph(),
			zeroMean.Dtype(),
			zeroMean.Dims(),
			G.WithValue(zeroMean),
			G.WithName("zeroMean"),
		)

		unitStddev := tensor.NewDense(
			tensor.Float64,
			n.stddev.Shape(),
			tensor.WithBacking(ones(size)),
		)
		n.unitStddev = G.NewTensor(
			n.stddev.Graph(),
			unitStddev.Dtype(),
			unitStddev.Dims(),
			G.WithValue(unitStddev),
			G.WithName("unitStddev"),
		)
		n.stdNormal = G.Must(NormalRand(n.mean, n.stddev, n.seed, samples))
		fmt.Println("SHAPES:", n.stdNormal.Shape(), n.stddev.Shape())
	}

	// Reparameterization trick
	// var out *G.Node
	// if samples > 1 {
	// 	out = G.Must(G.BroadcastHadamardProd(n.stdNormal, n.stddev, nil,
	// 		[]byte{0}))
	// 	out = G.Must(G.BroadcastAdd(out, n.mean, nil, []byte{0}))
	// } else {
	// 	out = G.Must(G.HadamardProd(n.stdNormal, n.stddev))
	// 	out = G.Must(G.Add(out, n.mean))
	// }

	// return out, nil
	return nil, nil
}

// TODO: Sample uses batch dim as dim 0, make all other functions
// TODO: use this as well
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
	// Normal always works with vectors. If an input meand or stddev
	// is given as a scalar, it is converted to a 1-vector.
	//
	// n.mean is always a vector. If n.mean.Shape() == []int{1}, then
	// it is analogous to a scalar. If n.mean.Shape()[0] > 1, then it
	// is a vector.
	if x.IsScalar() && n.mean.Shape()[0] == 1 {
		return G.Reshape(x, []int{1})

	} else if x.IsMatrix() && n.mean.Shape()[0] == 1 {
		if x.Shape()[0] == 1 {
			// x is a 1 x N matrix, reshape it into a vector
			return G.Reshape(x, []int{x.Shape()[1]})
		}
		return nil, fmt.Errorf("expected x to by a vector but got shape %v",
			x.Shape())

	} else if x.IsScalar() && n.mean.Shape()[0] > 1 {
		return nil, fmt.Errorf("expected x to be a vector or matrix but " +
			"got scalar")

	} else if x.IsMatrix() && n.mean.Shape()[0] > 1 &&
		x.Shape()[0] != n.mean.Shape()[0] {
		return nil, fmt.Errorf("expected x to have first dimension of "+
			"size %v but got shape %v", n.mean.Shape()[0],
			x.Shape())

	} else if len(x.Shape()) == 1 && n.mean.Shape()[0] == 1 {
		return x, nil

	} else if x.IsMatrix() && n.mean.Shape()[0] > 1 &&
		n.mean.Shape()[0] == x.Shape()[0] {
		return x, nil
	}

	return nil, fmt.Errorf("could not adjust shape of x expected shape "+
		"compatible with mean shape %v but got x shape %v", n.mean.Shape(),
		x.Shape())
}
