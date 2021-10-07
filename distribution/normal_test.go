package distribution

import (
	"math"
	rand "math/rand"
	"testing"
	"time"

	expRand "golang.org/x/exp/rand"

	"gonum.org/v1/gonum/stat/distuv"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestNormalVec(t *testing.T) {
	g := G.NewGraph()
	stddevT := tensor.NewDense(
		tensor.Float64,
		[]int{3},
		tensor.WithBacking([]float64{1, 1, 1}),
	)
	stddev := G.NewVector(
		g,
		stddevT.Dtype(),
		G.WithValue(stddevT),
	)

	meanT := tensor.NewDense(
		tensor.Float64,
		[]int{3},
		tensor.WithBacking([]float64{0, 0, 0}),
	)
	mean := G.NewVector(
		g,
		meanT.Dtype(),
		G.WithValue(meanT),
	)

	n, err := NewNormal(mean, stddev, uint64(11))
	if err != nil {
		t.Error(err)
	}

	xT := tensor.NewDense(
		tensor.Float64,
		[]int{3},
		tensor.WithBacking([]float64{1, 2, 3}),
	)
	x := G.NewVector(
		g,
		xT.Dtype(),
		G.WithValue(xT),
	)

	prob, err := n.Prob(x)
	if err != nil {
		t.Error(err)
	}
	var probVal G.Value
	G.Read(prob, &probVal)

	vm := G.NewTapeMachine(g)
	vm.RunAll()

	vm.Reset()
	vm.Close()
}

// TestNormalScalar tests the Prob function of the Normal struct with
// a scalar mean and standard deviation. All tests are completely
// randomized
func TestNormalProbScalar(t *testing.T) {
	const threshold float64 = 0.00001 // Threshold at which floats are equal
	const tests int = 30              // Number of tests to run
	rand.Seed(time.Now().UnixNano())

	// Set the scale for mean, stddev, and sampling
	meanScale := 2.
	stdScale := 2.

	// Min and Max number of dimensions for samples to compute the
	// PDF of
	const minSize = 1
	const maxSize = 10

	// Targets
	for i := 0; i < tests; i++ {
		// Random mean and stddev
		stddev := math.Exp(rand.Float64()) * stdScale
		mean := (rand.Float64() - 0.5) * meanScale
		dist := distuv.Normal{
			Mu:    mean,
			Sigma: stddev,
		}
		size := minSize + rand.Intn(maxSize-minSize)

		xBacking := make([]float64, size)
		probs := make([]float64, size)
		for j := range xBacking {
			xBacking[j] = dist.Rand()
			probs[j] = dist.Prob(xBacking[j])
		}

		g := G.NewGraph()
		stddevNode := G.NewScalar(g, tensor.Float64, G.WithName("stddev"))
		err := G.Let(stddevNode, stddev)
		if err != nil {
			t.Error(err)
		}

		meanNode := G.NewScalar(g, tensor.Float64, G.WithName("mean"))
		err = G.Let(meanNode, mean)
		if err != nil {
			t.Error(err)
		}

		n, err := NewNormal(meanNode, stddevNode, uint64(11))
		if err != nil {
			t.Error(err)
		}

		var x *G.Node
		if len(xBacking) == 1 {
			x = G.NewScalar(g, tensor.Float64)
			if err := G.Let(x, xBacking[0]); err != nil {
				t.Error(err)
			}
		} else {
			xT := tensor.NewDense(
				tensor.Float64,
				[]int{len(xBacking)},
				tensor.WithBacking(xBacking),
			)
			x = G.NewVector(
				g,
				xT.Dtype(),
				G.WithValue(xT),
			)
		}

		prob, err := n.Prob(x)
		if err != nil {
			t.Error(err)
		}
		var probVal G.Value
		G.Read(prob, &probVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		// Check output
		probOut := probVal.Data().([]float64)
		for j := range probOut {
			if math.Abs(probOut[j]-probs[j]) > threshold {
				t.Errorf("expected: %v received: %v for x: %v", probs[j],
					probOut[j], xBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

func TestVec(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal

	g := G.NewGraph()
	meanT := tensor.NewDense(
		tensor.Float64,
		[]int{3},
		tensor.WithBacking([]float64{1, 1, 1}),
	)
	mean := G.NewVector(g, meanT.Dtype(), G.WithValue(meanT))
	stddev := mean // ? Error with Gorgonia if we create separate stddev...

	n, err := NewNormal(mean, stddev, uint64(1))
	if err != nil {
		t.Error(err)
	}

	samplesBacking := []float64{0, 1, 2, 3, 4, 5}
	samplesT := tensor.NewDense(
		tensor.Float64,
		[]int{3, 2},
		tensor.WithBacking(samplesBacking),
	)
	samples := G.NewMatrix(g, tensor.Float64, G.WithValue(samplesT))

	prob, err := n.Prob(samples)
	if err != nil {
		t.Error(err)
	}
	var probVal G.Value
	G.Read(prob, &probVal)

	vm := G.NewTapeMachine(g)
	vm.RunAll()

	expected := []float64{0.24197072451914337, 0.3989422804014327,
		0.24197072451914337, 0.05399096651318806,
		0.0044318484119380075, 0.00013383022576488534,
	}
	probOut := probVal.Data().([]float64)

	for j := range probOut {
		if math.Abs(probOut[j]-expected[j]) > threshold {
			t.Errorf("expected: %v, received: %v, x: %x", expected[j],
				probOut[j], samplesBacking[j])
		}
	}

	vm.Reset()
	vm.Close()
}

// TestNormalEntropyScalar tests the Entropy() method of the Normal
// struct given scalar mean and standard deviation
func TestNormalEntropyScalar(t *testing.T) {
	const threshold float64 = 0.000001
	const tests int = 30
	const loc float64 = 3
	const scale float64 = 1.5

	for i := 0; i < tests; i++ {
		meanBacking := (rand.Float64() - 0.5) * loc
		stdBacking := math.Exp(rand.Float64()) * scale

		g := G.NewGraph()

		mean := G.NewScalar(g, tensor.Float64)
		err := G.Let(mean, meanBacking)
		if err != nil {
			t.Error(err)
		}

		stddev := G.NewScalar(g, tensor.Float64)
		err = G.Let(stddev, stdBacking)
		if err != nil {
			t.Error(err)
		}

		n, err := NewNormal(mean, stddev, uint64(1))
		if err != nil {
			t.Error(err)
		}

		entropy := n.Entropy()
		var eVal G.Value
		G.Read(entropy, &eVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		targetDist := distuv.Normal{
			Mu:    meanBacking,
			Sigma: stdBacking,
			Src:   expRand.NewSource(uint64(time.Now().UnixNano())),
		}

		if math.Abs(targetDist.Entropy()-
			eVal.Data().([]float64)[0]) > threshold {
			t.Errorf("expected: %v received: %v", targetDist.Entropy(),
				eVal.Data().([]float64)[0])
		}

		vm.Reset()
	}
}

// TestNormalEntropyVec tests the Entropy() method of the Normal
// struct given vector mean and standard deviation
func TestNormalEntropyVec(t *testing.T) {
	const threshold float64 = 0.000001
	const tests int = 15
	const scale float64 = 1.5

	const maxSize int = 32
	const minSize int = 1

	for i := 0; i < tests; i++ {
		size := minSize + rand.Intn(maxSize-minSize)

		meanBackings := make([]float64, size)
		stdBackings := make([]float64, size)
		entropyTarget := make([]float64, size)
		for j := range meanBackings {
			// ? Gorgonia keeps making mean and stddev point to the same
			// ? node... so I'll just give them the same value, I've logged
			// ? an issue on GitHub
			stdBackings[j] = math.Exp(rand.Float64()) * scale
			meanBackings[j] = stdBackings[j]
			targetDist := distuv.Normal{
				Mu:    meanBackings[j],
				Sigma: stdBackings[j],
				Src:   expRand.NewSource(uint64(time.Now().UnixNano())),
			}
			entropyTarget[j] = targetDist.Entropy()
		}

		g := G.NewGraph()

		meanT := tensor.New(
			tensor.WithShape(size),
			tensor.WithBacking(meanBackings),
		)
		mean := G.NewTensor(
			g,
			meanT.Dtype(),
			meanT.Dims(),
			G.WithShape(size),
			G.WithValue(meanT),
		)

		stddevT := tensor.New(
			tensor.WithShape(size),
			tensor.WithBacking(stdBackings),
		)
		stddev := G.NewTensor(
			g,
			stddevT.Dtype(),
			stddevT.Dims(),
			G.WithShape(size),
			G.WithValue(stddevT),
		)

		n, err := NewNormal(mean, stddev, uint64(1))
		if err != nil {
			t.Error(err)
		}

		entropy := n.Entropy()
		var eVal G.Value
		G.Read(entropy, &eVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		for j := range entropyTarget {
			if math.Abs(entropyTarget[j]-
				eVal.Data().([]float64)[j]) > threshold {
				t.Errorf("expected: %v received: %v", entropyTarget[j],
					eVal.Data().([]float64)[0])
			}
		}

		vm.Reset()
	}
}
