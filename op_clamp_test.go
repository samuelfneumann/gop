package gop

import (
	"math/rand"
	"testing"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestClamp(t *testing.T) {
	const numTests int = 15 // The number of random tests to run
	const scale float64 = 5 // Values are clamped based on scale

	// Randomly generated input has number of dimensions betwee dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 10
	const dimMin int = 1
	const dimMax int = 4
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTests; i++ {
		var min float64 = scale * (rand.Float64() - 1) // Random in [-scale, 0)
		var max float64 = scale * (rand.Float64())     // Random in [0, scale)

		// Construct the size of each dimension randomly, e.g. (3, 1, 2)
		size := randInt(dimMin+rand.Intn(dimMax-dimMin), sizeMin, sizeMax)

		// Get the total number of elements for the random input
		numElems := tensor.ProdInts(size)

		// Construct input data
		inTensor := tensor.NewDense(
			tensor.Float64,
			size,
			tensor.WithBacking(randF64(numElems, min*2, max*2)),
		)

		// Construct input node to be clamped
		g := G.NewGraph()
		in := G.NewTensor(
			g,
			tensor.Float64,
			len(inTensor.Shape()),
			G.WithValue(inTensor),
		)

		// Construct the clamp operation and save the outputted value
		c, err := Clamp(in, min, max)
		if err != nil {
			t.Error(err)
		}
		var cVal G.Value
		G.Read(c, &cVal)

		// Run the graph
		vm := G.NewTapeMachine(g)
		err = vm.RunAll()
		if err != nil {
			t.Error(err)
		}
		vm.Reset()

		// Ensure the output is clamped properly
		data := cVal.Data().([]float64)
		for i, elem := range data {
			if elem > max {
				t.Errorf("value at index %d greater than maximum %v", i, max)
			}
			if elem < min {
				t.Errorf("value at index %d less than minimum %v", i, min)
			}
		}
		vm.Close()
	}
}
