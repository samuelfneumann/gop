package gop

import (
	"math"
	"math/rand"
	"testing"

	"github.com/samuelfneumann/math32"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestErfinv_graph(t *testing.T) {
	const tolerance float64 = 0.0001
	const maxDims int = 5
	const minDims int = 2
	const maxDimSize int = 10

	shape := make([]int, minDims+rand.Intn(maxDims-minDims))
	for i := range shape {
		shape[i] = 1 + rand.Intn(maxDimSize-1) // Avoid dimension size 0
	}

	backing := make([]float64, tensor.ProdInts(shape))
	out := make([]float64, tensor.ProdInts(shape))
	grad := make([]float64, tensor.ProdInts(shape))
	for i := range backing {
		z := (rand.Float64() - 0.5) * 2.0
		backing[i] = z
		out[i] = math.Erfinv(backing[i])
		grad[i] = erfinvGrad(z) / float64(tensor.ProdInts(shape))
	}

	g := G.NewGraph()
	inTensor := tensor.NewDense(
		tensor.Float64,
		shape,
		tensor.WithBacking(backing),
	)

	in := G.NewTensor(
		g,
		tensor.Float64,
		len(shape),
		G.WithValue(inTensor),
	)
	computedNode, err := Erfinv(in)
	if err != nil {
		t.Error(err)
	}
	var computed G.Value
	G.Read(computedNode, &computed)

	// Ensure gradient can be computed
	mean := G.Must(G.Mean(computedNode))
	diff, err := G.Grad(mean, in)
	if err != nil {
		t.Error(err)
	}
	if len(diff) != 1 {
		t.Errorf("derivative should be a single node but got %v", len(diff))
	}
	var computedDiff G.Value
	G.Read(diff[0], &computedDiff)

	vm := G.NewTapeMachine(g)
	vm.RunAll()
	vm.Reset()

	// Check the output
	output := computed.Data().([]float64)
	for i := 0; i < len(out); i++ {
		if math.Abs(out[i]-output[i]) > tolerance {
			t.Errorf("incorrect value\nexpected: %v \nreceived:%v",
				out[i], output[i])
		}
	}

	// Check the gradient
	outGrad := computedDiff.Data().([]float64)
	for i := 0; i < len(out); i++ {
		if math.Abs(outGrad[i]-grad[i]) > tolerance {
			t.Errorf("incorrect gradient value\nexpected: %v \nreceived:%v",
				grad[i], outGrad[i])
		}
	}
}

func TestErfinvFloat64(t *testing.T) {
	erfinvDiff := erfinvDiffOp{}
	erfinv := newErfinvOp()
	const tolerance float64 = 0.0000001

	tests := 10
	for i := 0; i < tests; i++ {
		in := rand.Float64()
		out := math.Erfinv(in)
		outGrad := erfinvGrad(in)
		preGrad := rand.Float64()

		v, err := erfinv.Do(G.NewF64(in))
		if err != nil {
			t.Error(err)
		}

		if float64(*(v.(*G.F64))) != out {
			t.Errorf("incorret erfinv: expected %v received %v", out, v)
		}

		v, err = erfinvDiff.Do(G.NewF64(in), G.NewF64(preGrad))
		if err != nil {
			t.Error(err)
		}

		if math.Abs(float64(*(v.(*G.F64)))-(preGrad*outGrad)) > tolerance {
			t.Errorf("incorret erfinvDiff: expected %v received %v", outGrad, v)
		}
	}
}

func TestErfinvFloat32(t *testing.T) {
	erfinvDiff := erfinvDiffOp{}
	erfinv := newErfinvOp()
	const tolerance float32 = 0.0000001

	tests := 10
	for i := 0; i < tests; i++ {
		in := rand.Float32()
		out := math32.Erfinv(in)
		outGrad := (math32.Sqrt(math32.Pi) / 2.0) *
			math32.Exp(math32.Pow(math32.Erfinv(in), 2))
		preGrad := rand.Float32()

		v, err := erfinv.Do(G.NewF32(in))
		if err != nil {
			t.Error(err)
		}

		if float32(*(v.(*G.F32))) != out {
			t.Errorf("incorret erfinv: expected %v received %v", out, v)
		}

		v, err = erfinvDiff.Do(G.NewF32(in), G.NewF32(preGrad))
		if err != nil {
			t.Error(err)
		}

		if math32.Abs(float32(*(v.(*G.F32)))-(preGrad*outGrad)) > tolerance {
			t.Errorf("incorret erfinvDiff: expected %v received %v", outGrad, v)
		}
	}
}

func TestErfinv(t *testing.T) {
	erfinv := newErfinvOp()

	shapes := [][]int{
		{2, 2},
		{2, 2, 2},
		{2, 3, 5},
		{4, 3, 2, 1},
		{1},
	}
	for i := 0; i < len(shapes); i++ {
		inBacking := make([]float64, tensor.ProdInts(shapes[i]))
		outBacking := make([]float64, len(inBacking))
		for i := range outBacking {
			inBacking[i] = rand.Float64()
			outBacking[i] = math.Erfinv(inBacking[i])
		}
		in := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(inBacking),
		)
		inCheck := tensor.NewDense(
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
		v, err := erfinv.Do(in)
		if err != nil {
			t.Error(err)
		}

		// Ensure output is expected, input tensor modified, and
		// output shape is not changed
		if !v.(*tensor.Dense).Eq(out) {
			t.Errorf("expected: \n%v \nreceived: \n%v", out, v)
		} else if !inCheck.Eq(in) {
			t.Error("erfinv should not modify input value, but input modified")
		} else if !v.(*tensor.Dense).Shape().Eq(shapes[i]) {
			t.Errorf("erfinv should not modify shapes (%v modified to %v)",
				shapes[i], v.(*tensor.Dense).Shape())
		}
	}

	// Ensure Erfinv does not work with more than 1 input
	arityChecks := 10
	for i := 0; i < arityChecks; i++ {
		size := rand.Int()%9 + 2
		inputs := make([]G.Value, size)
		for i := range inputs {
			inputs[i] = G.NewF64(rand.Float64())
		}

		_, err := erfinv.Do(inputs...)
		if err == nil {
			t.Errorf("accepted %v inputs when Erfinv has arity of %v", len(inputs),
				erfinv.Arity())
		}
	}
}

func erfinvGrad(x float64) float64 {
	return (math.Sqrt(math.Pi) / 2.0) * math.Exp(math.Pow(math.Erfinv(x), 2))
}

func TestErfinvDiff(t *testing.T) {
	erfinvDiff := erfinvDiffOp{}

	shapes := [][]int{
		{2, 2},
		{2, 2, 2},
		{2, 3, 5},
		{4, 3, 2, 1},
		{1},
	}

	for i := 0; i < len(shapes); i++ {
		inBacking := make([]float64, tensor.ProdInts(shapes[i]))
		outBacking := make([]float64, len(inBacking))
		gradBacking := make([]float64, len(inBacking))
		for i := range outBacking {
			inBacking[i] = rand.Float64()
			gradBacking[i] = 0.1
			outBacking[i] = erfinvGrad(inBacking[i]) * gradBacking[i]
		}
		in := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(inBacking),
		)
		inCheck := tensor.NewDense(
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
		v, err := erfinvDiff.Do(in, grad)
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
		if !inCheck.Eq(in) {
			t.Error("erfinvDiff should not modify input value, but input " +
				"modified")
		} else if !v.(*tensor.Dense).Shape().Eq(shapes[i]) {
			t.Errorf("erfinvDiff should not modify shapes (%v modified to %v)",
				shapes[i], v.(*tensor.Dense).Shape())
		}
	}

}
