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

// TestNormalSampleScalar tests to ensure that consecutive runs of a
// graph on which Sample() has been called on a Normal with a scalar
// mean and stddev result in different values being sampled.
func TestNormalSampleScalar(t *testing.T) {
	const threshold float64 = 0.00001 // Threshold at which floats are equal
	const tests int = 30              // Number of tests to run
	rand.Seed(time.Now().UnixNano())

	// Set the scale for mean, stddev, and sampling
	meanScale := 2.
	stdScale := 2.

	const minBatchSize int = 1  // Minimum size of batch
	const maxBatchSize int = 50 // Maxium size of batch

	for i := 0; i < tests; i++ {
		// Random mean and stddev
		stddev := math.Exp(rand.Float64()) * stdScale
		mean := (rand.Float64() - 0.5) * meanScale

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

		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize) + 1
		sample, err := n.Sample(batchSize)
		if err != nil {
			t.Error(err)
		}
		var sampleVal G.Value
		G.Read(sample, &sampleVal)

		vm := G.NewTapeMachine(g)

		// Run through the graph once and record the sampled values
		vm.RunAll()
		sample1, err := G.CloneValue(sampleVal)
		if err != nil {
			t.Error(err)
		}
		vm.Reset()

		// Run through the graph again and record the next sampled values
		vm.RunAll()
		sample2, err := G.CloneValue(sampleVal)
		if err != nil {
			t.Error(err)
		}
		vm.Reset()

		// Ensure the sampled values are different
		sample1Data := sample1.(tensor.Tensor).Data().([]float64)
		sample2Data := sample2.(tensor.Tensor).Data().([]float64)
		for i := range sample1Data {
			// Assume sampled values are equal unless proved otherwise
			flag := true
			if math.Abs(sample1Data[i]-sample2Data[i]) > threshold {
				flag = false
			}

			if flag {
				t.Error("consecutive calls to Sample() resulted in the " +
					"same data sampled")
			}
		}

		vm.Close()
	}
}

// TestNormalSampleTensor tests to ensure that consecutive runs of a
// graph on which Sample() has been called on a Normal with a tensor
// mean and stddev result in different values being sampled.
func TestNormalSampleTensor(t *testing.T) {
	const threshold float64 = 0.000001 // Threshold to consider floats equal
	const tests int = 20               // Number of tests to run
	const scale float64 = 2.0

	const minSize int = 1       // Minimum number of dims in mean/stddev
	const maxSize int = 5       // Maximum number of dims in mean/stddev
	const minDimSize int = 1    // Minimum size of each dim in mean/stddev
	const maxDimSize int = 10   // Maximum size of each dim in mean/stddev
	const minBatchSize int = 1  // Minimum size of batch
	const maxBatchSize int = 10 // Maxium size of batch

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < tests; i++ {
		dims := minSize + rand.Intn(maxSize-minSize) + 1
		shape := randInt(dims, minDimSize, maxDimSize)

		numDists := tensor.ProdInts(shape)

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		g := G.NewGraph()
		meanT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(meanBacking),
		)
		mean := G.NewTensor(g, meanT.Dtype(), meanT.Dims(), G.WithValue(meanT),
			G.WithName("mean"))

		stddevT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(stddevBacking),
		)
		stddev := G.NewTensor(g, stddevT.Dtype(), stddevT.Dims(),
			G.WithValue(stddevT), G.WithName("stddev"))

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
		if err != nil {
			t.Error(err)
		}

		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize) + 1
		sample, err := n.Sample(batchSize)
		if err != nil {
			t.Error(err)
		}
		var sampleVal G.Value
		G.Read(sample, &sampleVal)

		vm := G.NewTapeMachine(g)

		// Run through the graph once and record the sampled values
		vm.RunAll()
		sample1, err := G.CloneValue(sampleVal)
		if err != nil {
			t.Error(err)
		}
		vm.Reset()

		// Run through the graph again and record the next sampled values
		vm.RunAll()
		sample2, err := G.CloneValue(sampleVal)
		if err != nil {
			t.Error(err)
		}
		vm.Reset()

		// Ensure the sampled values are different
		sample1Data := sample1.(tensor.Tensor).Data().([]float64)
		sample2Data := sample2.(tensor.Tensor).Data().([]float64)
		for i := range sample1Data {
			// Assume sampled values are equal unless proved otherwise
			flag := true
			if math.Abs(sample1Data[i]-sample2Data[i]) > threshold {
				flag = false
			}

			if flag {
				t.Error("consecutive calls to Sample() resulted in the " +
					"same data sampled")
			}
		}

		vm.Close()
	}
}

// TestNormalProbScalar tests the Prob method of the Normal struct
// on arbitrary, random scalar inputs. All tests are completely
// randomized.
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
		size := minSize + rand.Intn(maxSize-minSize+1)

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
			x = G.NewScalar(g, tensor.Float64, G.WithName("scalarX"))
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
				G.WithName("tensorX"),
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

// TestNormalProbVec tests the Prob method of the Normal struct
// on arbitrary, random vector inputs. All tests are completely
// randomized.
func TestNormalProbVec(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 15
	const scale float64 = 2.0

	const minNumDists int = 1
	const maxNumDists int = 10
	const minBatchSize int = 1
	const maxBatchSize int = 10

	for i := 0; i < tests; i++ {
		numDists := minNumDists + rand.Intn(maxNumDists-minNumDists+1)
		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize+1)
		size := []int{numDists}
		sampleSize := []int{batchSize, numDists}

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		sampleBacking := make([]float64, 0, numDists*batchSize)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				dist := distuv.Normal{
					Mu:    meanBacking[r],
					Sigma: stddevBacking[r],
					Src:   src,
				}
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

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
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
				t.Errorf("expected: %v, received: %v, x: %v", expected[j],
					probOut[j], sampleBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestNormalProbTensor tests the Prob method of the Normal struct
// on arbitrary, random tensor inputs. All tests are completely
// randomized.
func TestNormalProbTensor(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 5        // Number of tests to run
	const scale float64 = 2.0

	const minSize int = 1    // Minimum number of dims in mean/stddev
	const maxSize int = 5    // Maximum number of dims in mean/stddev
	const minDimSize int = 1 // Minimum size of each dim in mean/stddev
	const maxDimSize int = 5 // Maximum size of each dim in mean/stddev

	const minBatchSize int = 1  // Minimum number of samples in a batch
	const maxBatchSize int = 10 // Maximum number of samples in a batch

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < tests; i++ {
		dims := minSize + rand.Intn(maxSize-minSize) + 1
		shape := randInt(dims, minDimSize, maxDimSize)

		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize) + 1

		numDists := tensor.ProdInts(shape)
		sampleSize := append([]int{batchSize}, shape...)

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		sampleBacking := make([]float64, 0, numDists*batchSize)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				dist := distuv.Normal{
					Mu:    meanBacking[r],
					Sigma: stddevBacking[r],
					Src:   src,
				}
				sample := dist.Rand()
				sampleBacking = append(sampleBacking, sample)
				expected = append(expected, dist.Prob(sample))
			}
		}

		g := G.NewGraph()
		meanT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(meanBacking),
		)
		mean := G.NewTensor(g, meanT.Dtype(), meanT.Dims(), G.WithValue(meanT),
			G.WithName("mean"))

		stddevT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(stddevBacking),
		)
		stddev := G.NewTensor(g, stddevT.Dtype(), stddevT.Dims(),
			G.WithValue(stddevT), G.WithName("stddev"))

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
		if err != nil {
			t.Error(err)
		}

		samplesT := tensor.NewDense(
			tensor.Float64,
			sampleSize,
			tensor.WithBacking(sampleBacking),
		)
		samples := G.NewTensor(g, tensor.Float64, samplesT.Dims(),
			G.WithValue(samplesT), G.WithName("samples"))

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
				t.Errorf("expected: %v, received: %v, x: %v", expected[j],
					probOut[j], sampleBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestNormalLogProbScalar tests the LogProb method of the Normal struct
// on arbitrary, random scalar inputs. All tests are completely
// randomized.
func TestNormalLogProbScalar(t *testing.T) {
	const threshold float64 = 0.00001 // Threshold at which floats are equal
	const tests int = 15              // Number of tests to run
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
		size := minSize + rand.Intn(maxSize-minSize+1)

		xBacking := make([]float64, size)
		probs := make([]float64, size)
		for j := range xBacking {
			xBacking[j] = dist.Rand()
			probs[j] = dist.LogProb(xBacking[j])
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
			x = G.NewScalar(g, tensor.Float64, G.WithName("scalarX"))
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
				G.WithName("tensorX"),
			)
		}

		prob, err := n.LogProb(x)
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

// TestNormalLogProbVec tests the LogProb method of the Normal struct
// on arbitrary, random vector inputs. All tests are completely
// randomized.
func TestNormalLogProbVec(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 10
	const scale float64 = 2.0

	const minNumDists int = 1
	const maxNumDists int = 10
	const minBatchSize int = 1
	const maxBatchSize int = 10

	for i := 0; i < tests; i++ {
		numDists := minNumDists + rand.Intn(maxNumDists-minNumDists+1)
		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize+1)
		size := []int{numDists}
		sampleSize := []int{batchSize, numDists}

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		sampleBacking := make([]float64, 0, numDists*batchSize)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				dist := distuv.Normal{
					Mu:    meanBacking[r],
					Sigma: stddevBacking[r],
					Src:   src,
				}
				sample := dist.Rand()
				sampleBacking = append(sampleBacking, sample)
				expected = append(expected, dist.LogProb(sample))
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

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
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

		prob, err := n.LogProb(samples)
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

// TestNormalLogProbTensor tests the LogProb method of the Normal struct
// on arbitrary, random tensor inputs. All tests are completely
// randomized.
func TestNormalLogProbTensor(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 5        // Number of tests to run
	const scale float64 = 2.0

	const minSize int = 1    // Minimum number of dims in mean/stddev
	const maxSize int = 5    // Maximum number of dims in mean/stddev
	const minDimSize int = 1 // Minimum size of each dim in mean/stddev
	const maxDimSize int = 5 // Maximum size of each dim in mean/stddev

	const minBatchSize int = 1  // Minimum number of samples in a batch
	const maxBatchSize int = 10 // Maximum number of samples in a batch

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < tests; i++ {
		dims := minSize + rand.Intn(maxSize-minSize) + 1
		shape := randInt(dims, minDimSize, maxDimSize)

		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize) + 1

		numDists := tensor.ProdInts(shape)
		sampleSize := append([]int{batchSize}, shape...)

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		sampleBacking := make([]float64, 0, numDists*batchSize)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				dist := distuv.Normal{
					Mu:    meanBacking[r],
					Sigma: stddevBacking[r],
					Src:   src,
				}
				sample := dist.Rand()
				sampleBacking = append(sampleBacking, sample)
				expected = append(expected, dist.LogProb(sample))
			}
		}

		g := G.NewGraph()
		meanT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(meanBacking),
		)
		mean := G.NewTensor(g, meanT.Dtype(), meanT.Dims(), G.WithValue(meanT),
			G.WithName("mean"))

		stddevT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(stddevBacking),
		)
		stddev := G.NewTensor(g, stddevT.Dtype(), stddevT.Dims(),
			G.WithValue(stddevT), G.WithName("stddev"))

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
		if err != nil {
			t.Error(err)
		}

		samplesT := tensor.NewDense(
			tensor.Float64,
			sampleSize,
			tensor.WithBacking(sampleBacking),
		)
		samples := G.NewTensor(g, tensor.Float64, samplesT.Dims(),
			G.WithValue(samplesT), G.WithName("samples"))

		prob, err := n.LogProb(samples)
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
				t.Errorf("expected: %v, received: %v, x: %v", expected[j],
					probOut[j], sampleBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestNormalEntropyScalar tests the Entropy method of the Normal struct
// on arbitrary, random scalar inputs. All tests are completely
// randomized.
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

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
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

// TestNormalEntropyVec tests the Entropy method of the Normal struct
// on arbitrary, random vector inputs. All tests are completely
// randomized.
func TestNormalEntropyVec(t *testing.T) {
	// floats a and b are considered equal if |a-b| > threshold
	const threshold float64 = 0.000001

	const tests int = 15      // Number of tests to run
	const scale float64 = 1.5 // Scale at which to sample

	// Maximum size for mean and stddev vectors
	const maxSize int = 32
	const minSize int = 1

	for i := 0; i < tests; i++ {
		size := minSize + rand.Intn(maxSize-minSize+1)

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

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
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

// TestNormalEntropyTensor tests the Entropy method of the Normal struct
// on arbitrary, random tensor inputs. All tests are completely
// randomized.
func TestNormalEntropyTensor(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 5        // Number of tests to run
	const scale float64 = 2.0

	const minSize int = 1    // Minimum number of dims in mean/stddev
	const maxSize int = 5    // Maximum number of dims in mean/stddev
	const minDimSize int = 1 // Minimum size of each dim in mean/stddev
	const maxDimSize int = 5 // Maximum size of each dim in mean/stddev

	const minBatchSize int = 1  // Minimum number of samples in a batch
	const maxBatchSize int = 10 // Maximum number of samples in a batch

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < tests; i++ {
		dims := minSize + rand.Intn(maxSize-minSize) + 1
		shape := randInt(dims, minDimSize, maxDimSize)

		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize) + 1

		numDists := tensor.ProdInts(shape)

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				dist := distuv.Normal{
					Mu:    meanBacking[r],
					Sigma: stddevBacking[r],
					Src:   src,
				}
				expected = append(expected, dist.Entropy())
			}
		}

		g := G.NewGraph()
		meanT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(meanBacking),
		)
		mean := G.NewTensor(g, meanT.Dtype(), meanT.Dims(), G.WithValue(meanT),
			G.WithName("mean"))

		stddevT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(stddevBacking),
		)
		stddev := G.NewTensor(g, stddevT.Dtype(), stddevT.Dims(),
			G.WithValue(stddevT), G.WithName("stddev"))

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
		if err != nil {
			t.Error(err)
		}

		entropy := n.Entropy()
		var entropyVal G.Value
		G.Read(entropy, &entropyVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		entropyOut := entropyVal.Data().([]float64)

		for j := range entropyOut {
			if math.Abs(entropyOut[j]-expected[j]) > threshold {
				t.Errorf("expected: %v, received: %v", expected[j],
					entropyOut[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestNormalCdfScalar tests the Cdf method of the Normal struct
// on arbitrary, random scalar inputs. All tests are completely
// randomized.
func TestNormalCdfScalar(t *testing.T) {
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
		size := minSize + rand.Intn(maxSize-minSize+1)

		xBacking := make([]float64, size)
		probs := make([]float64, size)
		for j := range xBacking {
			xBacking[j] = dist.Rand()
			probs[j] = dist.CDF(xBacking[j])
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

		prob, err := n.Cdf(x)
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

// TestNormalCdfVec tests the Cdf method of the Normal struct
// on arbitrary, random vector inputs. All tests are completely
// randomized.
func TestNormalCdfVec(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 15
	const scale float64 = 2.0

	const minNumDists int = 1
	const maxNumDists int = 10
	const minBatchSize int = 1
	const maxBatchSize int = 10

	for i := 0; i < tests; i++ {
		numDists := minNumDists + rand.Intn(maxNumDists-minNumDists+1)
		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize+1)
		size := []int{numDists}
		sampleSize := []int{batchSize, numDists}

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		sampleBacking := make([]float64, 0, numDists*batchSize)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				dist := distuv.Normal{
					Mu:    meanBacking[r],
					Sigma: stddevBacking[r],
					Src:   src,
				}
				sample := dist.Rand()
				sampleBacking = append(sampleBacking, sample)
				expected = append(expected, dist.CDF(sample))
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

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
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

		prob, err := n.Cdf(samples)
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

// TestNormalCdfTensor tests the Cdf method of the Normal struct
// on arbitrary, random tensor inputs. All tests are completely
// randomized.
func TestNormalCdfTensor(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 5        // Number of tests to run
	const scale float64 = 2.0

	const minSize int = 1    // Minimum number of dims in mean/stddev
	const maxSize int = 5    // Maximum number of dims in mean/stddev
	const minDimSize int = 1 // Minimum size of each dim in mean/stddev
	const maxDimSize int = 5 // Maximum size of each dim in mean/stddev

	const minBatchSize int = 1  // Minimum number of samples in a batch
	const maxBatchSize int = 10 // Maximum number of samples in a batch

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < tests; i++ {
		dims := minSize + rand.Intn(maxSize-minSize) + 1
		shape := randInt(dims, minDimSize, maxDimSize)

		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize) + 1

		numDists := tensor.ProdInts(shape)
		sampleSize := append([]int{batchSize}, shape...)

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		sampleBacking := make([]float64, 0, numDists*batchSize)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				dist := distuv.Normal{
					Mu:    meanBacking[r],
					Sigma: stddevBacking[r],
					Src:   src,
				}
				sample := dist.Rand()
				sampleBacking = append(sampleBacking, sample)
				expected = append(expected, dist.CDF(sample))
			}
		}

		g := G.NewGraph()
		meanT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(meanBacking),
		)
		mean := G.NewTensor(g, meanT.Dtype(), meanT.Dims(), G.WithValue(meanT),
			G.WithName("mean"))

		stddevT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(stddevBacking),
		)
		stddev := G.NewTensor(g, stddevT.Dtype(), stddevT.Dims(),
			G.WithValue(stddevT), G.WithName("stddev"))

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
		if err != nil {
			t.Error(err)
		}

		samplesT := tensor.NewDense(
			tensor.Float64,
			sampleSize,
			tensor.WithBacking(sampleBacking),
		)
		samples := G.NewTensor(g, tensor.Float64, samplesT.Dims(),
			G.WithValue(samplesT), G.WithName("samples"))

		cdf, err := n.Cdf(samples)
		if err != nil {
			t.Error(err)
		}
		var cdfVal G.Value
		G.Read(cdf, &cdfVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		cdfOut := cdfVal.Data().([]float64)

		for j := range cdfOut {
			if math.Abs(cdfOut[j]-expected[j]) > threshold {
				t.Errorf("expected: %v, received: %v, x: %v", expected[j],
					cdfOut[j], sampleBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestNormalCdfinvScalar tests the Cdfinv method of the Normal struct
// on arbitrary, random scalar inputs. All tests are completely
// randomized.
func TestNormalCdfinvScalar(t *testing.T) {
	const threshold float64 = 0.00001 // Threshold at which floats are equal
	const tests int = 30              // Number of tests to run
	rand.Seed(time.Now().UnixNano())

	// Set the scale for mean, stddev, and sampling
	meanScale := 2.
	stdScale := 2.

	// Min and Maprob number of dimensions for samples to compute the
	// PDF of
	const minSize = 1
	const maprobSize = 10

	// Targets
	src := expRand.NewSource(uint64(time.Now().UnixNano()))
	dist := distuv.Uniform{
		Max: 1.0,
		Min: 0.0,
		Src: src,
	}
	for i := 0; i < tests; i++ {
		// Random mean and stddev
		stddev := math.Exp(rand.Float64()) * stdScale
		mean := (rand.Float64() - 0.5) * meanScale
		size := minSize + rand.Intn(maprobSize-minSize+1)

		probBacking := make([]float64, size)
		cdfs := make([]float64, size) // Correct CDFs
		for j := range probBacking {
			// Get a random probability to compute the inverse cdf of
			probBacking[j] = dist.Rand()

			// Store the correct CDF to ensure the calculation is correct
			cdfs[j] = mean + stddev*math.Sqrt(2.0)*math.Erfinv(
				2.0*probBacking[j]-1)
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

		var prob *G.Node
		if len(probBacking) == 1 {
			prob = G.NewScalar(g, tensor.Float64)
			if err := G.Let(prob, probBacking[0]); err != nil {
				t.Error(err)
			}
		} else {
			probT := tensor.NewDense(
				tensor.Float64,
				[]int{len(probBacking)},
				tensor.WithBacking(probBacking),
			)
			prob = G.NewVector(
				g,
				probT.Dtype(),
				G.WithValue(probT),
				G.WithName("probInput"),
			)
		}

		cdf, err := n.Cdfinv(prob)
		if err != nil {
			t.Error(err)
		}
		var cdfVal G.Value
		G.Read(cdf, &cdfVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		// Check output
		cdfOut := cdfVal.Data().([]float64)
		for j := range cdfOut {
			if math.Abs(cdfOut[j]-cdfs[j]) > threshold {
				t.Errorf("eprobpected: %v received: %v for prob: %v", cdfs[j],
					cdfOut[j], probBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestNormalCdfinvVec tests the Cdfinv method of the Normal struct
// on arbitrary, random vector inputs. All tests are completely
// randomized.
func TestNormalCdfinvVec(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 15       // Number of random tests to run
	const scale float64 = 2.0  // Scale of distributions' mean and stddev

	const minNumDists int = 1
	const maxNumDists int = 10
	const minBatchSize int = 1
	const maxBatchSize int = 10

	for i := 0; i < tests; i++ {
		numDists := minNumDists + rand.Intn(maxNumDists-minNumDists+1)
		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize+1)
		size := []int{numDists}
		sampleSize := []int{batchSize, numDists}

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		probBacking := make([]float64, 0, numDists*batchSize)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		dist := distuv.Uniform{
			Max: 1.0,
			Min: 0.0,
			Src: src,
		}
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				prob := dist.Rand()
				probBacking = append(probBacking, prob)

				icdf := meanBacking[r] + stddevBacking[r]*math.Sqrt(2.0)*
					math.Erfinv(2*prob-1)
				expected = append(expected, icdf)
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

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
		if err != nil {
			t.Error(err)
		}

		probT := tensor.NewDense(
			tensor.Float64,
			sampleSize,
			tensor.WithBacking(probBacking),
		)
		probs := G.NewMatrix(g, tensor.Float64, G.WithValue(probT),
			G.WithName("inputProbabilitiies"))

		quantile, err := n.Cdfinv(probs)
		if err != nil {
			t.Error(err)
		}
		var quantileVal G.Value
		G.Read(quantile, &quantileVal)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		quantileOut := quantileVal.Data().([]float64)

		for j := range quantileOut {
			if math.Abs(quantileOut[j]-expected[j]) > threshold {
				t.Errorf("expected: %v, received: %v, x: %x", expected[j],
					quantileOut[j], probBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestNormalCdfinvTensor tests the Cdfinv method of the Normal struct
// on arbitrary, random tensor inputs. All tests are completely
// randomized.
func TestNormalCdfinvTensor(t *testing.T) {
	const threshold = 0.000001 // Threshold for floats to be considered equal
	const tests int = 5        // Number of tests to run
	const scale float64 = 2.0

	const minSize int = 1    // Minimum number of dims in mean/stddev
	const maxSize int = 5    // Maximum number of dims in mean/stddev
	const minDimSize int = 1 // Minimum size of each dim in mean/stddev
	const maxDimSize int = 5 // Maximum size of each dim in mean/stddev

	const minBatchSize int = 1  // Minimum number of samples in a batch
	const maxBatchSize int = 10 // Maximum number of samples in a batch

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < tests; i++ {
		dims := minSize + rand.Intn(maxSize-minSize) + 1
		shape := randInt(dims, minDimSize, maxDimSize)

		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize) + 1

		numDists := tensor.ProdInts(shape)
		sampleSize := append([]int{batchSize}, shape...)

		meanBacking := make([]float64, numDists)
		stddevBacking := make([]float64, numDists)
		probBacking := make([]float64, 0, numDists*batchSize)
		expected := make([]float64, 0, numDists*batchSize)
		src := expRand.NewSource(uint64(time.Now().UnixNano()))
		dist := distuv.Uniform{
			Max: 1.0,
			Min: 0.0,
			Src: src,
		}
		for r := 0; r < numDists; r++ {
			mean := (rand.Float64() - 0.5) * scale
			stddev := math.Exp(rand.Float64() * scale)
			meanBacking[r] = mean
			stddevBacking[r] = stddev
		}

		for c := 0; c < batchSize; c++ {
			for r := 0; r < numDists; r++ {
				prob := dist.Rand()
				probBacking = append(probBacking, prob)

				icdf := meanBacking[r] + stddevBacking[r]*math.Sqrt(2.0)*
					math.Erfinv(2*prob-1)
				expected = append(expected, icdf)
			}
		}

		g := G.NewGraph()
		meanT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(meanBacking),
		)
		mean := G.NewTensor(g, meanT.Dtype(), meanT.Dims(), G.WithValue(meanT),
			G.WithName("mean"))

		stddevT := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(stddevBacking),
		)
		stddev := G.NewTensor(g, stddevT.Dtype(), stddevT.Dims(),
			G.WithValue(stddevT), G.WithName("stddev"))

		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
		if err != nil {
			t.Error(err)
		}

		probsT := tensor.NewDense(
			tensor.Float64,
			sampleSize,
			tensor.WithBacking(probBacking),
		)
		probs := G.NewTensor(g, tensor.Float64, probsT.Dims(),
			G.WithValue(probsT), G.WithName("inputProbabilities"))

		quantile, err := n.Cdfinv(probs)
		if err != nil {
			t.Error(err)
		}
		var quantileVals G.Value
		G.Read(quantile, &quantileVals)

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		quantileOut := quantileVals.Data().([]float64)

		for j := range quantileOut {
			if math.Abs(quantileOut[j]-expected[j]) > threshold {
				t.Errorf("expected: %v, received: %v, x: %v", expected[j],
					quantileOut[j], probBacking[j])
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// func TestNormalRsampleVec(t *testing.T) {
// 	const tests int = 1       // Number of random tests to run
// 	const scale float64 = 2.0 // Scale of distributions' mean and stddev

// 	const minRows int = 1
// 	const maxRows int = 10

// 	for i := 0; i < tests; i++ {
// 		rows := minRows + rand.Intn(maxRows-minRows+1)
// 		size := []int{rows}

// 		meanBacking := make([]float64, rows)
// 		stddevBacking := make([]float64, rows)
// 		for r := 0; r < rows; r++ {
// 			mean := (rand.Float64() - 0.5) * scale
// 			stddev := math.Exp(rand.Float64() * scale)
// 			meanBacking[r] = mean
// 			stddevBacking[r] = stddev
// 		}

// 		g := G.NewGraph()
// 		meanT := tensor.NewDense(
// 			tensor.Float64,
// 			size,
// 			tensor.WithBacking(meanBacking),
// 		)
// 		mean := G.NewVector(g, meanT.Dtype(), G.WithValue(meanT),
// 			G.WithName("mean"))

// 		stddevT := tensor.NewDense(
// 			tensor.Float64,
// 			size,
// 			tensor.WithBacking(stddevBacking),
// 		)
// 		stddev := G.NewVector(g, stddevT.Dtype(), G.WithValue(stddevT),
// 			G.WithName("stddev"))

// 		n, err := NewNormal(mean, stddev, uint64(time.Now().UnixNano()))
// 		if err != nil {
// 			t.Error(err)
// 		}

// 		sample, err := n.Rsample(2)
// 		if err != nil {
// 			t.Error(err)
// 		}
// 		var sampleVal G.Value
// 		G.Read(sample, &sampleVal)

// 		vm := G.NewTapeMachine(g)
// 		vm.RunAll()

// 		fmt.Println(stddev.Value())
// 		fmt.Println(mean.Value())
// 		fmt.Println(sampleVal)

// 		vm.Reset()
// 		vm.Close()
// 	}
// }
