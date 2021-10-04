package gop

import (
	"math/rand"
	"testing"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestF64Clamp(t *testing.T) {
	const numTests int = 15     // The number of random tests to run
	const clipScale float64 = 2 // Legal ranges generated based on clipScale
	const scale float64 = 5     // Values are clamped based on scale

	// Randomly generated input has number of dimensions betwee dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 10
	const dimMin int = 1
	const dimMax int = 4
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTests; i++ {
		var min float64 = clipScale * (rand.Float64() - 1) // [-clipScale, 0)
		var max float64 = clipScale * rand.Float64()       // [0, clipScale)

		// Construct the size of each dimension randomly, e.g. (3, 1, 2)
		size := randInt(dimMin+rand.Intn(dimMax-dimMin), sizeMin, sizeMax)

		// Get the total number of elements for the random input
		numElems := tensor.ProdInts(size)

		// Construct input data
		inBacking := randF64(numElems, min*scale, max*scale)
		inTensor := tensor.NewDense(
			tensor.Float64,
			size,
			tensor.WithBacking(inBacking),
		)

		// Construct the target/correct gradient
		gradTarget := make([]float64, numElems)
		for i := range inBacking {
			if inBacking[i] > min && inBacking[i] < max {
				gradTarget[i] = 1
			}

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
		c, err := Clamp(in, min, max, false)
		if err != nil {
			t.Error(err)
		}
		var cVal G.Value
		G.Read(c, &cVal)

		// Construct loss + gradient
		loss := G.Must(G.Sum(c))
		grad, err := G.Grad(loss, in)
		if err != nil {
			t.Error(err)
		}
		if len(grad) != 1 {
			t.Errorf("expected 1 gradient got %v", len(grad))
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

		// Ensure the gradient was calculated correctly
		gradData := gradVal.Data().([]float64)
		for i := range gradTarget {
			if gradTarget[i] != gradData[i] {
				t.Errorf("expected: %v \nreceived: %v \nindex: %d",
					gradTarget[i], gradData[i], i)
			}
		}

		vm.Close()
	}
}

func TestF32Clamp(t *testing.T) {
	const numTests int = 15     // The number of random tests to run
	const clipScale float32 = 2 // Legal ranges generated based on clipScale
	const scale float32 = 5     // Values are clamped based on scale

	// Randomly generated input has number of dimensions betwee dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 10
	const dimMin int = 1
	const dimMax int = 4
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTests; i++ {
		var min float32 = clipScale * (rand.Float32() - 1) // [-clipScale, 0)
		var max float32 = clipScale * rand.Float32()       // [0, clipScale)

		// Construct the size of each dimension randomly, e.g. (3, 1, 2)
		size := randInt(dimMin+rand.Intn(dimMax-dimMin), sizeMin, sizeMax)

		// Get the total number of elements for the random input
		numElems := tensor.ProdInts(size)

		// Construct input data
		inBacking := randF32(numElems, min*scale, max*scale)
		inTensor := tensor.NewDense(
			tensor.Float32,
			size,
			tensor.WithBacking(inBacking),
		)

		// Construct the target/correct gradient
		gradTarget := make([]float32, numElems)
		for i := range inBacking {
			if inBacking[i] > min && inBacking[i] < max {
				gradTarget[i] = 1
			}

		}

		// Construct input node to be clamped
		g := G.NewGraph()
		in := G.NewTensor(
			g,
			tensor.Float32,
			len(inTensor.Shape()),
			G.WithValue(inTensor),
		)

		// Construct the clamp operation and save the outputted value
		c, err := Clamp(in, min, max, false)
		if err != nil {
			t.Error(err)
		}
		var cVal G.Value
		G.Read(c, &cVal)

		// Construct loss + gradient
		loss := G.Must(G.Sum(c))
		grad, err := G.Grad(loss, in)
		if err != nil {
			t.Error(err)
		}
		if len(grad) != 1 {
			t.Errorf("expected 1 gradient got %v", len(grad))
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

		// Ensure the output is clamped properly
		data := cVal.Data().([]float32)
		for i, elem := range data {
			if elem > max {
				t.Errorf("value at index %d greater than maximum %v", i, max)
			}
			if elem < min {
				t.Errorf("value at index %d less than minimum %v", i, min)
			}
		}

		// Ensure the gradient was calculated correctly
		gradData := gradVal.Data().([]float32)
		for i := range gradTarget {
			if gradTarget[i] != gradData[i] {
				t.Errorf("expected: %v \nreceived: %v \nindex: %d",
					gradTarget[i], gradData[i], i)
			}
		}

		vm.Close()
	}
}

func TestIntClamp(t *testing.T) {
	t.Log("cannot take gradient of integer tensor, testing forward pass only")

	const numTests int = 15  // The number of random tests to run
	const clipScale int = 20 // Legal ranges generated based on clipScale
	const scale int = 2      // Values are clamped based on scale

	// Randomly generated input has number of dimensions betwee dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 10
	const dimMin int = 1
	const dimMax int = 4
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTests; i++ {
		var min int = -rand.Intn(clipScale)
		var max int = rand.Intn(clipScale)

		// Construct the size of each dimension randomly, e.g. (3, 1, 2)
		size := randInt(dimMin+rand.Intn(dimMax-dimMin), sizeMin, sizeMax)

		// Get the total number of elements for the random input
		numElems := tensor.ProdInts(size)

		// Construct input data
		inBacking := randInt(numElems, min*scale, max*scale)
		inTensor := tensor.NewDense(
			tensor.Int,
			size,
			tensor.WithBacking(inBacking),
		)

		// Construct input node to be clamped
		g := G.NewGraph()
		in := G.NewTensor(
			g,
			tensor.Int,
			len(inTensor.Shape()),
			G.WithValue(inTensor),
		)

		// Construct the clamp operation and save the outputted value
		c, err := Clamp(in, min, max, false)
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
		data := cVal.Data().([]int)
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
