package gop

import (
	"math"
	"math/rand"
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func prod(ints ...int) int {
	i := 1
	for _, elem := range ints {
		i *= elem
	}
	return i
}

func TestErf(t *testing.T) {
	erf := newErfOp()

	shapes := [][]int{
		{2, 2},
		{2, 2, 2},
		{2, 3, 5},
		{4, 3, 2, 1},
		{1},
	}
	for i := 0; i < len(shapes); i++ {
		inBacking := make([]float64, prod(shapes[i]...))
		outBacking := make([]float64, len(inBacking))
		for i := range outBacking {
			inBacking[i] = rand.Float64()
			outBacking[i] = math.Erf(inBacking[i])
		}
		in := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(inBacking),
		)

		out := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(outBacking),
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

func erfGrad(x float64) float64 {
	return (2 / math.Sqrt(math.Pi)) * math.Exp(-math.Pow(x, 2))
}

func TestErfDiff(t *testing.T) {
	erfDiff := erfDiffOp{}

	shapes := [][]int{
		{2, 2},
		{2, 2, 2},
		{2, 3, 5},
		{4, 3, 2, 1},
		{1},
	}

	for i := 0; i < len(shapes); i++ {
		inBacking := make([]float64, prod(shapes[i]...))
		outBacking := make([]float64, len(inBacking))
		gradBacking := make([]float64, len(inBacking))
		for i := range outBacking {
			inBacking[i] = rand.Float64()
			gradBacking[i] = 0.1
			outBacking[i] = erfGrad(inBacking[i]) * gradBacking[i]
		}
		in := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(inBacking),
		)
		out := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(outBacking),
		)
		grad := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(gradBacking),
		)

		// Run the operation
		v, err := erfDiff.Do(in, grad)
		if err != nil {
			t.Error(err)
		}

		// Ensure output is correct to within tolerance
		const tolerance float64 = 0.000001
		for i := range out.Data().([]float64) {
			diff := math.Abs(out.Data().([]float64)[i] -
				v.Data().([]float64)[i])

			if diff > tolerance {
				t.Errorf("expected: %v \nreceived: %v \nat index %d",
					out.Data().([]float64)[i], v.Data().([]float64)[i], i)
			}
		}

		// Ensure shape is correct and input not modified
		if v.(*tensor.Dense).Eq(in) {
			t.Error("erfDiff should not modify input value, but input " +
				"modified")
		} else if !v.(*tensor.Dense).Shape().Eq(shapes[i]) {
			t.Errorf("erfDiff should not modify shapes (%v modified to %v)",
				shapes[i], v.(*tensor.Dense).Shape())
		}
	}

}
