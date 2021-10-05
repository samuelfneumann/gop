package gop

import (
	"math"
	"math/rand"
	"testing"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestRepeat(t *testing.T) {
	const numTests int = 15 // The number of random tests to run
	const maxRepeats int = 10
	const threshold float64 = 0.00001 // Threshold to determine floats equal

	// Randomly generated input has number of dimensions between dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 5
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

		// Construct the gradient target
		gradTarget := make([]float64, inTensor.Size())
		for i := range gradTarget {
			gradTarget[i] = 1.0 / float64(len(gradTarget))
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

		// Construct loss + gradient
		loss := G.Must(G.Mean(c))
		grad, err := G.Grad(loss, in)
		if err != nil {
			t.Error(err)
		}
		if len(grad) != 1 {
			t.Errorf("expected 1 gradient node, received %v", len(grad))
		}
		var gradVal G.Value
		G.Read(grad[0], &gradVal)

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

		gradData, ok := gradVal.Data().([]float64)
		if !ok {
			// Gradient has a single value
			gradData = []float64{gradVal.Data().(float64)}
		}
		for i := range gradData {
			if math.Abs(gradData[i]-gradTarget[i]) > threshold {
				coords, err := tensor.Itol(i, gradVal.Shape(),
					gradVal.(*tensor.Dense).Strides())

				if err != nil {
					t.Errorf("error is computed gradient \nexpected: %v"+
						"\nreceived: %v \ncoords unknown due to error: %v \n",
						gradData, gradTarget, err)
				}

				t.Errorf("error is computed gradient \nexpected: %v"+
					"\nreceived: %v \nerror at coords:%v \n",
					gradData, gradTarget, coords)
			}
		}
		vm.Close()
	}
}
