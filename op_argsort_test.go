package gop

import (
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestArgsortVM(t *testing.T) {
	in := []*tensor.Dense{
		tensor.NewDense(
			tensor.Float64,
			[]int{3, 3},
			tensor.WithBacking([]float64{1, 4, 3, 6, 8, 3, 5, 2, 1}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{10},
			tensor.WithBacking([]float64{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Float32,
			[]int{10},
			tensor.WithBacking([]float32{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Float32,
			[]int{1},
			tensor.WithBacking([]float32{9}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{1},
			tensor.WithBacking([]float64{91.1}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{0},
			tensor.WithBacking([]float64{}),
		),
	}

	out := []*tensor.Dense{
		tensor.NewDense(
			tensor.Int,
			[]int{3, 3},
			tensor.WithBacking([]int{0, 2, 1, 2, 0, 1, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{10},
			tensor.WithBacking([]int{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{10},
			tensor.WithBacking([]int{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{1},
			tensor.WithBacking([]int{0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{1},
			tensor.WithBacking([]int{0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{0},
			tensor.WithBacking([]int{}),
		),
	}
	axis := []int{1, 0, 0, 1, 0, 0}
	errorExpected := []bool{false, false, false, true, false, true}

	for i := range in {
		g := G.NewGraph()
		input := G.NewTensor(
			g,
			in[i].Dtype(),
			in[i].Shape().Dims(),
			G.WithValue(in[i]),
		)

		argSorted, err := Argsort(input, axis[i])
		if err != nil {
			t.Error(err)
		}

		var sorted G.Value
		G.Read(argSorted, &sorted)

		vm := G.NewTapeMachine(g)

		err = vm.RunAll()
		if err != nil {
			if !errorExpected[i] {
				t.Error(err)
			}
		} else {
			if !sorted.(*tensor.Dense).Eq(out[i]) {
				t.Errorf("expected: \n%v \nreceived: \n%v", out[i], sorted)
			}
		}

		vm.Reset()
		vm.Close()
	}
}

func TestArgsort(t *testing.T) {
	in := []*tensor.Dense{
		tensor.NewDense(
			tensor.Float64,
			[]int{3, 3},
			tensor.WithBacking([]float64{1, 4, 3, 6, 8, 3, 5, 2, 1}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{10},
			tensor.WithBacking([]float64{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Float32,
			[]int{10},
			tensor.WithBacking([]float32{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Float32,
			[]int{1},
			tensor.WithBacking([]float32{9}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{1},
			tensor.WithBacking([]float64{91.1}),
		),
		tensor.NewDense(
			tensor.Float64,
			[]int{0},
			tensor.WithBacking([]float64{}),
		),
	}

	out := []*tensor.Dense{
		tensor.NewDense(
			tensor.Int,
			[]int{3, 3},
			tensor.WithBacking([]int{0, 2, 1, 2, 0, 1, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{10},
			tensor.WithBacking([]int{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{10},
			tensor.WithBacking([]int{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{1},
			tensor.WithBacking([]int{0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{1},
			tensor.WithBacking([]int{0}),
		),
		tensor.NewDense(
			tensor.Int,
			[]int{0},
			tensor.WithBacking([]int{}),
		),
	}
	axis := []int{1, 0, 0, 1, 0, 0}
	errorExpected := []bool{false, false, false, true, false, true}

	for i := range in {
		argsort := newArgsortOp(axis[i], in[i].Shape().Dims())
		sorted, err := argsort.Do(in[i])
		if err != nil {
			if !errorExpected[i] {
				t.Error(err)
			}
		} else {
			if !sorted.(*tensor.Dense).Eq(out[i]) {
				t.Errorf("expected: \n%v \nreceived: \n%v", out[i], sorted)
			}
		}
	}
}
