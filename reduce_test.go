package gop

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestReduceAdd(t *testing.T) {
	// Get a random shape for the tensor to squeeze
	var shape []int = []int{2, 1, 1}
	// fmt.Println("Shape:", shape)

	// Get a random axis to squeeze
	axis := 0
	// fmt.Println("AXIS:", axis)

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
		G.WithShape(shape...),
	)
	computedNode, err := ReduceAdd(in, axis)
	if err != nil {
		t.Error(err)
	}
	var computed G.Value
	G.Read(computedNode, &computed)

	// // Draw graph
	// b, err := dot.Marshal(g)
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println(string(b))

	// Run the graph
	vm := G.NewTapeMachine(g)
	err = vm.RunAll()
	if err != nil {
		t.Error(err)
	}

	fmt.Println("INPUT:\n", inTensor)
	fmt.Println(inTensor.Shape())
	fmt.Println()
	fmt.Println("COMPUTED:\n", computed)
	fmt.Println(computedNode.Shape())
	fmt.Println(computed.Shape())
	fmt.Println(axis)

	vm.Reset()
	vm.Close()
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
		if len(inTensor.Shape()) == 1 && inTensor.Shape()[0] == 1 {
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
