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

func TestNormalProbVecRandom(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 10
	const scale float64 = 2.0

	const minRows int = 1
	const maxRows int = 10
	const minCols int = 1
	const maxCols int = 10

	for i := 0; i < tests; i++ {
		rows := minRows + rand.Intn(maxRows-minRows)
		cols := minCols + rand.Intn(maxCols-minCols)
		size := []int{rows}
		sampleSize := []int{rows, cols}

		meanBacking := make([]float64, rows)
		stddevBacking := make([]float64, rows)
		sampleBacking := make([]float64, 0, rows*cols)
		expected := make([]float64, 0, rows*cols)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		for r := 0; r < rows; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev

			dist := distuv.Normal{
				Mu:    mean,
				Sigma: stddev,
				Src:   src,
			}

			for c := 0; c < cols; c++ {
				sample := dist.Rand()
				sampleBacking = append(sampleBacking, sample)
				expected = append(expected, dist.Prob(sample))
			}
		}

		g := G.NewGraph()
		meanT := tensor.NewDense(
			tensor.Float64,
			size,
			tensor.WithBacking(meanBacking),
		)
		mean := G.NewVector(g, meanT.Dtype(), G.WithValue(meanT),
			G.WithName("mean"))

		stddevT := tensor.NewDense(
			tensor.Float64,
			size,
			tensor.WithBacking(stddevBacking),
		)
		stddev := G.NewVector(g, stddevT.Dtype(), G.WithValue(stddevT),
			G.WithName("stddev"))

		n, err := NewNormal(mean, stddev, uint64(1))
		if err != nil {
			t.Error(err)
		}

		samplesT := tensor.NewDense(
			tensor.Float64,
			sampleSize,
			tensor.WithBacking(sampleBacking),
		)
		samples := G.NewMatrix(g, tensor.Float64, G.WithValue(samplesT),
			G.WithName("samples"))

		prob, err := n.Prob(samples)
		if err != nil {
			t.Error(err)
		}
		var probVal G.Value
		G.Read(prob, &probVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		probOut := probVal.Data().([]float64)

		for j := range probOut {
			if math.Abs(probOut[j]-expected[j]) > threshold {
				t.Errorf("expected: %v, received: %v, x: %x", expected[j],
					probOut[j], sampleBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

func TestNormalProbVec(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal

	meanBacking := [][]float64{
		{0, 0, 0},
		{-1, 0.1, 1},
	}

	stddevBacking := [][]float64{
		{1, 1, 1},
		{1, 3, 5},
	}

	size := [][]int{
		{3},
		{3},
	}

	sampleBacking := [][]float64{
		{0, 1, 2, 3, 4, 5},
		{
			-0.45394790295001486, 1.7454617691515626, 8.884286881614553,
			-1.7893568667493487, 2.1221545782377915, -0.6835244227267903,
			-1.6683640237354735, -1.4242390100872995, 10.961879493597525,
			-2.0412423041012464, 0.894317394736687, 5.377979125764084,
			1.2284993556522985, 1.6448287587878656, 7.519768718499081,
			-1.188418845764601, 2.6410849879245064, -8.590253781506744,
			-0.2540761081210544, 6.85247735326013, -2.829389115247283,
		},
	}

	sampleSize := [][]int{
		{3, 2},
		{3, 7},
	}

	expected := [][]float64{
		{
			3.98942280e-01, 2.41970725e-01,
			5.39909665e-02, 4.43184841e-03,
			1.33830226e-04, 1.48671951e-06,
		},
		{
			3.43686636e-01, 9.20766770e-03, 2.43116346e-22, 2.92152117e-01,
			3.04923927e-03, 3.79455887e-01, 3.19086272e-01, 1.16878155e-01,
			1.89330018e-04, 1.03078080e-01, 1.28400236e-01, 2.82923036e-02,
			1.23897407e-01, 1.16468558e-01, 3.40977415e-02, 7.25006330e-02,
			7.56044871e-02, 1.26786501e-02, 7.73178383e-02, 4.02193816e-02,
			5.95070274e-02,
		},
	}

	for i := 0; i < len(meanBacking); i++ {
		g := G.NewGraph()
		meanT := tensor.NewDense(
			tensor.Float64,
			size[i],
			tensor.WithBacking(meanBacking[i]),
		)
		mean := G.NewVector(g, meanT.Dtype(), G.WithValue(meanT),
			G.WithName("mean"))

		stddevT := tensor.NewDense(
			tensor.Float64,
			size[i],
			tensor.WithBacking(stddevBacking[i]),
		)
		stddev := G.NewVector(g, stddevT.Dtype(), G.WithValue(stddevT),
			G.WithName("stddev"))

		n, err := NewNormal(mean, stddev, uint64(1))
		if err != nil {
			t.Error(err)
		}

		samplesT := tensor.NewDense(
			tensor.Float64,
			sampleSize[i],
			tensor.WithBacking(sampleBacking[i]),
		)
		samples := G.NewMatrix(g, tensor.Float64, G.WithValue(samplesT),
			G.WithName("samples"))

		prob, err := n.Prob(samples)
		if err != nil {
			t.Error(err)
		}
		var probVal G.Value
		G.Read(prob, &probVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		probOut := probVal.Data().([]float64)

		for j := range probOut {
			if math.Abs(probOut[j]-expected[i][j]) > threshold {
				t.Errorf("expected: %v, received: %v, x: %x", expected[i][j],
					probOut[j], sampleBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
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
	// floats a and b are considered equal if |a-b| > threshold
	const threshold float64 = 0.000001

	const tests int = 15      // Number of tests to run
	const scale float64 = 1.5 // Scale at which to sample

	// Maximum size for mean and stddev vectors
	const maxSize int = 32
	const minSize int = 1

	for i := 0; i < tests; i++ {
		size := minSize + rand.Intn(maxSize-minSize)

		meanBackings := make([]float64, size)
		stdBackings := make([]float64, size)
		entropyTarget := make([]float64, size)
		for j := range meanBackings {
			stdBackings[j] = math.Exp(rand.Float64()) * scale
			meanBackings[j] = rand.Float64() * scale
			targetDist := distuv.Normal{
				Mu:    meanBackings[j],
				Sigma: stdBackings[j],
				Src:   expRand.NewSource(uint64(time.Now().UnixNano())),
			}
			entropyTarget[j] = targetDist.Entropy()
		}

		g := G.NewGraph(G.WithGraphName("graphNormalEntropyTest"))

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
			G.WithName("mean"),
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
			G.WithName("stddev"),
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
