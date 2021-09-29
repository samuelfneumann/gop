package gop_test

import (
	"math/rand"
	"testing"

	"github.com/samuelfneumann/gop"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestErf(t *testing.T) {
	erf := gop.NewErfOp()

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
	shapes := []tensor.Shape{
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
		} else if !v.(*tensor.Dense).Shape().Eq(shapes[i]) {
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

func TestErfDiff(t *testing.T) {
	erfDiff := gop.ErfDiffOp{}

	grads := []*tensor.Dense{
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2},
			tensor.WithBacking([]float64{1, 1, 1, 1}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2},
			tensor.WithBacking([]float64{0.1, 0.1, 0.1, 0.1}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2, 2},
			tensor.WithBacking([]float64{1, 1, 1, 1, 1, 1, 1, 1}),
		),
	}

	ins := []*tensor.Dense{
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2},
			tensor.WithBacking([]float64{-1, 0, 1, 2}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2},
			tensor.WithBacking([]float64{-1, 0, 1, 2}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2, 2},
			tensor.WithBacking([]float64{-0.01, -0.1, 0.0, 0.1, 0.01, 0.07,
				0.008, 0.91}),
		),
	}

	targets := []*tensor.Dense{
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2},
			tensor.WithBacking([]float64{
				0.4151074974205947,
				1.1283791670955126,
				0.4151074974205947,
				0.020666985354092053,
			}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2},
			tensor.WithBacking([]float64{
				0.041510749742059476,
				0.11283791670955126,
				0.041510749742059476,
				0.0020666985354092053,
			}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{2, 2, 2},
			tensor.WithBacking([]float64{
				1.1282663348205109,
				1.1171516067889369,
				1.1283791670955126,
				1.1171516067889369,
				1.1282663348205109,
				1.122863633270276,
				1.1283069531396897,
				0.4929646741550179,
			}),
		),
	}

	for i := range targets {
		out, err := erfDiff.Do(ins[i], grads[i])
		if err != nil {
			t.Errorf("could not compute gradient: %v", err)
		}

		// Ensure output is expected, input tensor modified, and
		// output shape is not changed
		if !out.(*tensor.Dense).Eq(targets[i]) {
			t.Errorf("expected: \n%v \nreceived: \n%v", targets[i], out)
		} else if out.(*tensor.Dense).Eq(ins[i]) {
			t.Error("erfDiff should not modify input value, but input " +
				"modified")
		} else if !out.(*tensor.Dense).Shape().Eq(ins[i].Shape()) {
			t.Errorf("erf should not modify shapes (%v modified to %v)",
				ins[i].Shape(), out.(*tensor.Dense).Shape())
		}
	}

}
