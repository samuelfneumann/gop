package gop

import (
	"math"
	"math/rand"
	"testing"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TestReduceMean tests the ReduceMean function with keepdims == true
func TestReduceMean(t *testing.T) {
	// Test parameters
	rand.Seed(time.Now().UnixNano())

	const threshold float64 = 0.00001 // Threshold to consider floats equal
	const tests int = 20              // Number of tests to run

	const maxDims int = 6     // Maximum number of tensor dimensions to test on
	const minDims int = 1     // Minimum number of tensor dimensions to test on
	const maxDimSize int = 10 // Maximum number of elements per dimension

	for i := 0; i < tests; i++ {
		// Get a random shape for the tensor to squeeze
		shape := make([]int, minDims+rand.Intn(maxDims-minDims))
		for i := range shape {
			shape[i] = 1 + rand.Intn(maxDimSize-1)
		}

		// Get a random axis to squeeze
		axis := rand.Intn(len(shape))

		// Create the input tensor
		backing := make([]float64, tensor.ProdInts(shape))
		for i := range backing {
			z := (rand.Float64() - 0.5) * 2.0 // ∈ [-1, 1)
			backing[i] = z
		}
		inTensor := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(backing),
		)

		// Calculate target shape
		targetShape := inTensor.Shape().Clone()[:axis]
		if axis != inTensor.Dims()-1 {
			targetShape = append(targetShape, inTensor.Shape()[axis+1:]...)
		}

		// Calculate target
		ind := make([]tensor.Slice, inTensor.Dims())
		n := float64(inTensor.Shape()[axis])
		target := tensor.NewDense(inTensor.Dtype(), targetShape)
		for row := 0; row < inTensor.Shape()[axis]; row++ {
			// Get the next row to compute on
			ind[axis] = G.S(row)
			nextRowView, err := inTensor.Slice(ind...)
			if err != nil {
				t.Error(err)
			}

			// Get the next row and reshape it, which is required since
			// if there any any dimensions of 1 in the input Tensor,
			// they will be squeezed, which we don't want to happen
			// in the target
			nextRow := nextRowView.Materialize().(*tensor.Dense)
			err = nextRow.Reshape(targetShape...)
			if err != nil {
				t.Error(err)
			}

			// Update the target
			target, err = target.Add(nextRow)
			if err != nil {
				t.Error(err)
			}
		}
		mean, err := tensor.Div(target, n)
		if err != nil {
			t.Error(err)
		}
		target = mean.(*tensor.Dense)

		// Create the computational graph
		g := G.NewGraph()

		// Create input tensor
		in := G.NewTensor(
			g,
			tensor.Float64,
			len(shape),
			G.WithValue(inTensor),
			G.WithShape(shape...),
		)

		// Set the operation to test
		computedNode, err := ReduceMean(in, axis, true)
		if err != nil {
			t.Error(err)
		}
		var computed G.Value
		G.Read(computedNode, &computed)

		// Run the graph
		vm := G.NewTapeMachine(g)
		err = vm.RunAll()
		if err != nil {
			t.Error(err)
		}

		// Enusre the output has the correct shape
		if !target.Shape().Eq(computed.Shape()) {
			t.Errorf("expected shape: %v \nreceived shape: %v\n", target.Shape(),
				computed.Shape())
		}

		// Ensure the output has the correct values computed
		if target.Dims() != 0 {
			targetBacking := target.Data().([]float64)
			computedBacking := computed.Data().([]float64)
			for i := range targetBacking {
				if math.Abs(targetBacking[i]-computedBacking[i]) > threshold {
					t.Errorf("incorrect result computed \n\texpected: %v "+
						"\n\treceived: %v\n", targetBacking[i], computedBacking[i])
				}
			}
		} else {
			targetBacking := target.Data().(float64)
			computedBacking := computed.Data().(float64)
			if math.Abs(targetBacking-computedBacking) > threshold {
				t.Errorf("incorrect result computed \n\texpected: %v "+
					"\n\treceived: %v\n", targetBacking, computedBacking)
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestReduceAdd tests the ReduceAdd function with keepdims == true
func TestReduceAdd(t *testing.T) {
	// Test parameters
	rand.Seed(time.Now().UnixNano())

	const threshold float64 = 0.00001 // Threshold to consider floats equal
	const tests int = 20              // Number of tests to run

	const maxDims int = 6     // Maximum number of tensor dimensions to test on
	const minDims int = 1     // Minimum number of tensor dimensions to test on
	const maxDimSize int = 10 // Maximum number of elements per dimension

	for i := 0; i < tests; i++ {
		// Get a random shape for the tensor to squeeze
		shape := make([]int, minDims+rand.Intn(maxDims-minDims))
		for i := range shape {
			shape[i] = 1 + rand.Intn(maxDimSize-1)
		}

		// Get a random axis to squeeze
		axis := rand.Intn(len(shape))

		// Create the input tensor
		backing := make([]float64, tensor.ProdInts(shape))
		for i := range backing {
			z := (rand.Float64() - 0.5) * 2.0 // ∈ [-1, 1)
			backing[i] = z
		}
		inTensor := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(backing),
		)

		// Calculate target shape
		targetShape := inTensor.Shape().Clone()[:axis]
		if axis != inTensor.Dims()-1 {
			targetShape = append(targetShape, inTensor.Shape()[axis+1:]...)
		}

		// Calculate target
		ind := make([]tensor.Slice, inTensor.Dims())
		target := tensor.NewDense(inTensor.Dtype(), targetShape)
		for row := 0; row < inTensor.Shape()[axis]; row++ {
			// Get the next row to compute on
			ind[axis] = G.S(row)
			nextRowView, err := inTensor.Slice(ind...)
			if err != nil {
				t.Error(err)
			}

			// Get the next row and reshape it, which is required since
			// if there any any dimensions of 1 in the input Tensor,
			// they will be squeezed, which we don't want to happen
			// in the target
			nextRow := nextRowView.Materialize().(*tensor.Dense)
			err = nextRow.Reshape(targetShape...)
			if err != nil {
				t.Error(err)
			}

			// Update the target
			target, err = target.Add(nextRow)
			if err != nil {
				t.Error(err)
			}
		}

		// Create the computational graph
		g := G.NewGraph()

		// Create input tensor
		in := G.NewTensor(
			g,
			tensor.Float64,
			len(shape),
			G.WithValue(inTensor),
			G.WithShape(shape...),
		)

		// Set the operation to test
		computedNode, err := ReduceAdd(in, axis, true)
		if err != nil {
			t.Error(err)
		}
		var computed G.Value
		G.Read(computedNode, &computed)

		// Run the graph
		vm := G.NewTapeMachine(g)
		err = vm.RunAll()
		if err != nil {
			t.Error(err)
		}

		// Enusre the output has the correct shape
		if !target.Shape().Eq(computed.Shape()) {
			t.Errorf("expected shape: %v \nreceived shape: %v\n", target.Shape(),
				computed.Shape())
		}

		// Ensure the output has the correct values computed
		if target.Dims() != 0 {
			targetBacking := target.Data().([]float64)
			computedBacking := computed.Data().([]float64)
			for i := range targetBacking {
				if math.Abs(targetBacking[i]-computedBacking[i]) > threshold {
					t.Errorf("incorrect result computed \n\texpected: %v "+
						"\n\treceived: %v\n", targetBacking[i], computedBacking[i])
				}
			}
		} else {
			targetBacking := target.Data().(float64)
			computedBacking := computed.Data().(float64)
			if math.Abs(targetBacking-computedBacking) > threshold {
				t.Errorf("incorrect result computed \n\texpected: %v "+
					"\n\treceived: %v\n", targetBacking, computedBacking)
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestReduceSub tests the ReduceSub function with keepdims == true
func TestReduceSub(t *testing.T) {
	// Test parameters
	rand.Seed(time.Now().UnixNano())

	const threshold float64 = 0.00001 // Threshold to consider floats equal
	const tests int = 20              // Number of tests to run

	const maxDims int = 6     // Maximum number of tensor dimensions to test on
	const minDims int = 1     // Minimum number of tensor dimensions to test on
	const maxDimSize int = 10 // Maximum number of elements per dimension

	for i := 0; i < tests; i++ {
		// Get a random shape for the tensor to squeeze
		shape := make([]int, minDims+rand.Intn(maxDims-minDims))
		for i := range shape {
			shape[i] = 1 + rand.Intn(maxDimSize-1)
		}

		// Get a random axis to squeeze
		axis := rand.Intn(len(shape))

		// Create the input tensor
		backing := make([]float64, tensor.ProdInts(shape))
		for i := range backing {
			z := (rand.Float64() - 0.5) * 2.0 // ∈ [-1, 1)
			backing[i] = z
		}
		inTensor := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(backing),
		)

		// Calculate target shape
		targetShape := inTensor.Shape().Clone()[:axis]
		if axis != inTensor.Dims()-1 {
			targetShape = append(targetShape, inTensor.Shape()[axis+1:]...)
		}

		// === Calculate the target ===
		// Use the first row as a starting point for target
		ind := make([]tensor.Slice, inTensor.Dims())
		ind[axis] = G.S(0)
		targetView, err := inTensor.Slice(ind...)
		if err != nil {
			t.Error(err)
		}

		// Materialize the row and reshape it to the proper shape
		target := targetView.Materialize().(*tensor.Dense)
		err = target.Reshape(targetShape...)
		if err != nil {
			t.Error(err)
		}

		// Run the function to test on target and next row for all next
		// rows
		for row := 1; row < inTensor.Shape()[axis]; row++ {
			// Get the next row to compute on
			ind[axis] = G.S(row)
			nextRowView, err := inTensor.Slice(ind...)
			if err != nil {
				t.Error(err)
			}

			// Get the next row and reshape it, which is required since
			// if there any any dimensions of 1 in the input Tensor,
			// they will be squeezed, which we don't want to happen
			// in the target
			nextRow := nextRowView.Materialize().(*tensor.Dense)
			err = nextRow.Reshape(targetShape...)
			if err != nil {
				t.Error(err)
			}

			// Update the target
			target, err = target.Sub(nextRow)
			if err != nil {
				t.Error(err)
			}
		}

		// Create the computational graph
		g := G.NewGraph()

		// Create input tensor
		in := G.NewTensor(
			g,
			tensor.Float64,
			len(shape),
			G.WithValue(inTensor),
			G.WithShape(shape...),
		)

		// Set the operation to test
		computedNode, err := ReduceSub(in, axis, true)
		if err != nil {
			t.Error(err)
		}
		var computed G.Value
		G.Read(computedNode, &computed)

		// Run the graph
		vm := G.NewTapeMachine(g)
		err = vm.RunAll()
		if err != nil {
			t.Error(err)
		}

		// Enusre the output has the correct shape
		if !target.Shape().Eq(computed.Shape()) {
			t.Errorf("expected shape: %v \nreceived shape: %v\n", target.Shape(),
				computed.Shape())
		}

		// Ensure the output has the correct values computed
		if target.Dims() != 0 {
			targetBacking := target.Data().([]float64)
			computedBacking := computed.Data().([]float64)
			for i := range targetBacking {
				if math.Abs(targetBacking[i]-computedBacking[i]) > threshold {
					t.Errorf("incorrect result computed \n\texpected: %v "+
						"\n\treceived: %v\n", targetBacking[i], computedBacking[i])
				}
			}
		} else {
			targetBacking := target.Data().(float64)
			computedBacking := computed.Data().(float64)
			if math.Abs(targetBacking-computedBacking) > threshold {
				t.Errorf("incorrect result computed \n\texpected: %v "+
					"\n\treceived: %v\n", targetBacking, computedBacking)
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestReduceProd tests the ReduceProd function with keepdims == true
func TestReduceProd(t *testing.T) {
	// Test parameters
	rand.Seed(time.Now().UnixNano())

	const threshold float64 = 0.00001 // Threshold to consider floats equal
	const tests int = 20              // Number of tests to run

	const maxDims int = 6     // Maximum number of tensor dimensions to test on
	const minDims int = 1     // Minimum number of tensor dimensions to test on
	const maxDimSize int = 10 // Maximum number of elements per dimension

	for i := 0; i < tests; i++ {
		// Get a random shape for the tensor to squeeze
		shape := make([]int, minDims+rand.Intn(maxDims-minDims))
		for i := range shape {
			shape[i] = 1 + rand.Intn(maxDimSize-1)
		}

		// Get a random axis to squeeze
		axis := rand.Intn(len(shape))

		// Create the input tensor
		backing := make([]float64, tensor.ProdInts(shape))
		for i := range backing {
			z := (rand.Float64() - 0.5) * 2.0 // ∈ [-1, 1)
			backing[i] = z
		}
		inTensor := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(backing),
		)

		// Calculate target shape
		targetShape := inTensor.Shape().Clone()[:axis]
		if axis != inTensor.Dims()-1 {
			targetShape = append(targetShape, inTensor.Shape()[axis+1:]...)
		}

		// === Calculate the target ===
		// Use the first row as a starting point for target
		ind := make([]tensor.Slice, inTensor.Dims())
		ind[axis] = G.S(0)
		targetView, err := inTensor.Slice(ind...)
		if err != nil {
			t.Error(err)
		}

		// Materialize the row and reshape it to the proper shape
		target := targetView.Materialize().(*tensor.Dense)
		err = target.Reshape(targetShape...)
		if err != nil {
			t.Error(err)
		}

		// Run the function to test on target and next row for all next
		// rows
		for row := 1; row < inTensor.Shape()[axis]; row++ {
			// Get the next row to compute on
			ind[axis] = G.S(row)
			nextRowView, err := inTensor.Slice(ind...)
			if err != nil {
				t.Error(err)
			}

			// Get the next row and reshape it, which is required since
			// if there any any dimensions of 1 in the input Tensor,
			// they will be squeezed, which we don't want to happen
			// in the target
			nextRow := nextRowView.Materialize().(*tensor.Dense)
			err = nextRow.Reshape(targetShape...)
			if err != nil {
				t.Error(err)
			}

			// Update the target
			target, err = target.Mul(nextRow)
			if err != nil {
				t.Error(err)
			}
		}

		// Create the computational graph
		g := G.NewGraph()

		// Create input tensor
		in := G.NewTensor(
			g,
			tensor.Float64,
			len(shape),
			G.WithValue(inTensor),
			G.WithShape(shape...),
		)

		// Set the operation to test
		computedNode, err := ReduceProd(in, axis, true)
		if err != nil {
			t.Error(err)
		}
		var computed G.Value
		G.Read(computedNode, &computed)

		// Run the graph
		vm := G.NewTapeMachine(g)
		err = vm.RunAll()
		if err != nil {
			t.Error(err)
		}

		// Enusre the output has the correct shape
		if !target.Shape().Eq(computed.Shape()) {
			t.Errorf("expected shape: %v \nreceived shape: %v\n", target.Shape(),
				computed.Shape())
		}

		// Ensure the output has the correct values computed
		if target.Dims() != 0 {
			targetBacking := target.Data().([]float64)
			computedBacking := computed.Data().([]float64)
			for i := range targetBacking {
				if math.Abs(targetBacking[i]-computedBacking[i]) > threshold {
					t.Errorf("incorrect result computed \n\texpected: %v "+
						"\n\treceived: %v\n", targetBacking[i], computedBacking[i])
				}
			}
		} else {
			targetBacking := target.Data().(float64)
			computedBacking := computed.Data().(float64)
			if math.Abs(targetBacking-computedBacking) > threshold {
				t.Errorf("incorrect result computed \n\texpected: %v "+
					"\n\treceived: %v\n", targetBacking, computedBacking)
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestReduceDiv tests the ReduceDiv function with keepdims == true
func TestReduceDiv(t *testing.T) {
	// Test parameters
	rand.Seed(time.Now().UnixNano())

	const threshold float64 = 0.00001 // Threshold to consider floats equal
	const tests int = 20              // Number of tests to run

	const maxDims int = 6     // Maximum number of tensor dimensions to test on
	const minDims int = 1     // Minimum number of tensor dimensions to test on
	const maxDimSize int = 10 // Maximum number of elements per dimension

	for i := 0; i < tests; i++ {
		// Get a random shape for the tensor to squeeze
		shape := make([]int, minDims+rand.Intn(maxDims-minDims))
		for i := range shape {
			shape[i] = 1 + rand.Intn(maxDimSize-1)
		}

		// Get a random axis to squeeze
		axis := rand.Intn(len(shape))

		// Create the input tensor
		backing := make([]float64, tensor.ProdInts(shape))
		for i := range backing {
			z := (rand.Float64() - 0.5) * 2.0 // ∈ [-1, 1)
			backing[i] = z
		}
		inTensor := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(backing),
		)

		// Calculate target shape
		targetShape := inTensor.Shape().Clone()[:axis]
		if axis != inTensor.Dims()-1 {
			targetShape = append(targetShape, inTensor.Shape()[axis+1:]...)
		}

		// === Calculate the target ===
		// Use the first row as a starting point for target
		ind := make([]tensor.Slice, inTensor.Dims())
		ind[axis] = G.S(0)
		targetView, err := inTensor.Slice(ind...)
		if err != nil {
			t.Error(err)
		}

		// Materialize the row and reshape it to the proper shape
		target := targetView.Materialize().(*tensor.Dense)
		err = target.Reshape(targetShape...)
		if err != nil {
			t.Error(err)
		}

		// Run the function to test on target and next row for all next
		// rows
		for row := 1; row < inTensor.Shape()[axis]; row++ {
			// Get the next row to compute on
			ind[axis] = G.S(row)
			nextRowView, err := inTensor.Slice(ind...)
			if err != nil {
				t.Error(err)
			}

			// Get the next row and reshape it, which is required since
			// if there any any dimensions of 1 in the input Tensor,
			// they will be squeezed, which we don't want to happen
			// in the target
			nextRow := nextRowView.Materialize().(*tensor.Dense)
			err = nextRow.Reshape(targetShape...)
			if err != nil {
				t.Error(err)
			}

			// Update the target
			target, err = target.Div(nextRow)
			if err != nil {
				t.Error(err)
			}
		}

		// Create the computational graph
		g := G.NewGraph()

		// Create input tensor
		in := G.NewTensor(
			g,
			tensor.Float64,
			len(shape),
			G.WithValue(inTensor),
			G.WithShape(shape...),
		)

		// Set the operation to test
		computedNode, err := ReduceDiv(in, axis, true)
		if err != nil {
			t.Error(err)
		}
		var computed G.Value
		G.Read(computedNode, &computed)

		// Run the graph
		vm := G.NewTapeMachine(g)
		err = vm.RunAll()
		if err != nil {
			t.Error(err)
		}

		// Enusre the output has the correct shape
		if !target.Shape().Eq(computed.Shape()) {
			t.Errorf("expected shape: %v \nreceived shape: %v\n", target.Shape(),
				computed.Shape())
		}

		// Ensure the output has the correct values computed
		if target.Dims() != 0 {
			targetBacking := target.Data().([]float64)
			computedBacking := computed.Data().([]float64)
			for i := range targetBacking {
				if math.Abs(targetBacking[i]-computedBacking[i]) > threshold {
					t.Errorf("incorrect result computed \n\texpected: %v "+
						"\n\treceived: %v\n", targetBacking[i], computedBacking[i])
				}
			}
		} else {
			targetBacking := target.Data().(float64)
			computedBacking := computed.Data().(float64)
			if math.Abs(targetBacking-computedBacking) > threshold {
				t.Errorf("incorrect result computed \n\texpected: %v "+
					"\n\treceived: %v\n", targetBacking, computedBacking)
			}
		}

		vm.Reset()
		vm.Close()
	}
}

// TestSqueeze tests the squeeze function
func TestSqueeze(t *testing.T) {
	const tolerance float64 = 0.00001
	const tests int = 100

	const maxDims int = 10
	const minDims int = 1
	const maxDimSize int = 3
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < tests; i++ {
		// Get a random shape for the tensor to squeeze
		shape := make([]int, minDims+rand.Intn(maxDims-minDims))
		for i := range shape {
			shape[i] = 1 + rand.Intn(maxDimSize-1) // Avoid dimension size 0
		}

		// Get a random axis to squeeze
		axis := rand.Intn(len(shape))

		// Create the backing tensor
		backing := make([]float64, tensor.ProdInts(shape))
		for i := range backing {
			z := (rand.Float64() - 0.5) * 2.0
			backing[i] = z
		}
		inTensor := tensor.NewDense(
			tensor.Float64,
			shape,
			tensor.WithBacking(backing),
		)

		// Create the computational graph
		g := G.NewGraph()

		in := G.NewTensor(
			g,
			tensor.Float64,
			len(shape),
			G.WithValue(inTensor),
		)
		computedNode, err := Squeeze(in, axis)
		if err != nil {
			t.Error(err)
		}
		var computed G.Value
		G.Read(computedNode, &computed)

		// Run the graph
		vm := G.NewTapeMachine(g)
		vm.RunAll()
		vm.Reset()

		// Check the output shape
		outShape := computed.Shape()
		if shape[axis] != 1 && !outShape.Eq(shape) {
			t.Errorf("expected: %v\n received: %v\n", shape, outShape)
		} else if shape[axis] == 1 {
			target := inTensor.Shape()[:axis]
			if axis < len(shape)-1 {
				target = append(target, shape[axis+1:]...)
			}
			if !target.Eq(outShape) {
				t.Errorf("expected: %v\n received: %v\n", target, outShape)
			}
		}

		// Check that the output was unmodified
		if (len(inTensor.Shape()) == 1 && inTensor.Shape()[0] == 1) ||
			inTensor.Dims() == 0 {
			outBacking := computed.Data().(float64)
			if math.Abs(outBacking-backing[0]) > tolerance {
				t.Errorf("input data was modified")
			}
		} else {
			outBacking := computed.Data().([]float64)
			for i := range outBacking {
				if math.Abs(outBacking[i]-backing[i]) > tolerance {
					t.Errorf("input data was modified")
				}
			}
		}

		vm.Close()
	}
}
