package gop

import (
	"math/rand"
	"testing"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestRepeat(t *testing.T) {
	const numTests int = 15 // The number of random tests to run
	const maxRepeats int = 10

	// Randomly generated input has number of dimensions between dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 10
	const dimMin int = 1
	const dimMax int = 4
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTests; i++ {
		// Construct the size of each dimension randomly, e.g. (3, 1, 2)
		size := randInt(dimMin+rand.Intn(dimMax-dimMin), sizeMin, sizeMax)

		// Construct the axis to repeat along as well as the number of
		// repeats to do
		axis := rand.Intn(len(size))
		repeats := rand.Intn(maxRepeats) + 1

		// Get the total number of elements for the random input
		numElems := tensor.ProdInts(size)

		// Construct input data
		inBacking := randF64(numElems, -1., 1.)
		inTensor := tensor.NewDense(
			tensor.Float64,
			size,
			tensor.WithBacking(inBacking),
		)

		// Construct the target/correct gradient
		repeatTarget, err := tensor.Repeat(inTensor, axis, repeats)
		if err != nil {
			t.Error(err)
		}

		// Construct input node to be clamped
		g := G.NewGraph()
		in := G.NewTensor(
			g,
			tensor.Float64,
			len(inTensor.Shape()),
			G.WithValue(inTensor),
		)

		// Construct the clamp operation and save the outputted value
		c, err := Repeat(in, axis, repeats)
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

		if !cVal.(tensor.Tensor).Eq(repeatTarget) {
			t.Errorf("expected: \n%v \nreceived: \n%v\n", repeatTarget, cVal)
		}

		vm.Close()
	}
}
