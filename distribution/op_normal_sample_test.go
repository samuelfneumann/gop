package distribution

import (
	"math"
	rand "math/rand"
	"testing"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TestNormalRand tests to ensure that the node returned by NormalRand
// returns different sampled data on consecutive runs of the
// computational graph
func TestNormalRand(t *testing.T) {
	const threshold float64 = 0.00000001 // Threshold to consider floats equal
	const tests int = 50                 // Number of tests to run
	const scale float64 = 2.0
	const stdOffset float64 = 0.001

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
			stddev := (math.Exp(rand.Float64()) + stdOffset) * scale
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

		batchSize := minBatchSize + rand.Intn(maxBatchSize-minBatchSize) + 1
		s, err := NormalRand(mean, stddev, uint64(time.Now().UnixNano()),
			batchSize)
		if err != nil {
			t.Error(err)
		}
		var sampled G.Value
		G.Read(s, &sampled)

		vm := G.NewTapeMachine(g)
		vm.RunAll()
		sampled1, err := G.CloneValue(sampled)
		if err != nil {
			t.Error(err)
		}
		vm.Reset()
		vm.RunAll()

		sampled1Data := sampled1.Data().([]float64)
		sampled2Data := sampled.Data().([]float64)
		for i := range sampled1Data {
			// Assume we saw the same sample on both runs through graph
			// unless prove otherwise
			flag := true
			if math.Abs(sampled1Data[i]-sampled2Data[i]) > threshold {
				flag = false
			}

			if flag {
				t.Error("sampled the same data twice")
			}
		}

		vm.Reset()
		vm.Close()
	}
}
