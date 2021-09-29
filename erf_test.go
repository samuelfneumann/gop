package op_test

import (
	"math/rand"
	"testing"

	"github.com/samuelfneumann/op"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestErf(t *testing.T) {
	erf := op.NewErfOp()

	// Create input tensors
	inBackings := [][]float64{
		{
			-2, 0, 1, 2,
		},
		{
			-4, -3, -0.1, -0.01, 0.01, 0.1, 3, 4,
		},
	}

	// Create the target tensors
	outBackings := [][]float64{
		{
			-0.9953222650189527,
			0.0,
			0.8427007929497149,
			0.9953222650189527,
		},
		{
			-0.9999999845827421,
			-0.9999779095030014,
			-0.1124629160182849,
			-0.011283415555849618,
			0.011283415555849618,
			0.1124629160182849,
			0.9999779095030014,
			0.9999999845827421,
		},
	}

	// Shapes for each of the input/output/target tensors
	shapes := [][]int{
		{2, 2},
		{2, 2, 2},
	}

	for i := range inBackings {
		in := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(inBackings[i]),
		)

		out := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(outBackings[i]),
		)

		// Run the operation
		v, err := erf.Do(in)
		if err != nil {
			t.Error(err)
		}

		// Ensure output is expected, input tensor modified, and
		// output shape is not changed
		if !v.(*tensor.Dense).Eq(out) {
			t.Errorf("expected: \n%v \nreceived: \n%v", out, v)
		} else if !v.(*tensor.Dense).Eq(in) {
			t.Error("erf should modify input value, but input left unmodified")
		} else if !v.(*tensor.Dense).Shape().Eq(out.Shape()) {
			t.Errorf("erf should not modify shapes (%v modified to %v)",
				shapes[i], v.(*tensor.Dense).Shape())
		}
	}

	// Ensure Erf does not work with more than 1 input
	arityChecks := 10
	for i := 0; i < arityChecks; i++ {
		size := rand.Int()%9 + 2
		inputs := make([]G.Value, size)
		for i := range inputs {
			inputs[i] = G.NewF64(rand.Float64())
		}

		_, err := erf.Do(inputs...)
		if err == nil {
			t.Errorf("accepted %v inputs when Erf has arity of %v", len(inputs),
				erf.Arity())
		}
	}
}
