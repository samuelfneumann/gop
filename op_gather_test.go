package gop

import (
	"fmt"
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGather(t *testing.T) {
	fmt.Println("No gradient computed for gather... is it because indices is int?")
	inBacking := [][]float64{
		{0, 1, 2, 3, 4, 5, 6, 7},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
			18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
	}
	inShapes := [][]int{
		{1, 4, 2},
		{8, 4},
	}

	indicesBacking := [][]int{
		{0, 2, 0, 1},
		{2, 3, 1, 1, 2, 1, 3, 3, 1, 1},
	}
	indicesShapes := [][]int{
		{1, 2, 2},
		{5, 2},
	}

	axis := []int{1, 0}

	outBacking := [][]float64{
		{0, 5, 0, 3},
		{8, 13, 4, 5, 8, 5, 12, 13, 4, 5},
		{2, 3, 5, 5, 10, 9, 15, 15, 17, 17},
	}

	for i := range inBacking {
		inT := tensor.NewDense(
			tensor.Float64,
			inShapes[i],
			tensor.WithBacking(inBacking[i]),
		)

		out := tensor.NewDense(
			tensor.Float64,
			indicesShapes[i],
			tensor.WithBacking(outBacking[i]),
		)

		indicesT := tensor.NewDense(
			tensor.Int,
			indicesShapes[i],
			tensor.WithBacking(indicesBacking[i]),
		)

		g := G.NewGraph()
		in := G.NewTensor(
			g,
			inT.Dtype(),
			inT.Shape().Dims(),
			G.WithValue(inT),
		)
		indices := G.NewTensor(
			g,
			indicesT.Dtype(),
			indicesT.Shape().Dims(),
			G.WithValue(indicesT),
		)

		pred, err := Gather(in, axis[i], indices)
		if err != nil {
			t.Error(err)
		}
		var predVal G.Value
		G.Read(pred, &predVal)

		// Loss and gradient
		loss := G.Must(G.Mean(pred))
		_, err = G.Grad(loss, in)
		if err != nil {
			t.Error(err)
		}
		// if len(grad) != 1 {
		// 	t.Errorf("expected 1 gradient got %v", len(grad))
		// 	continue
		// }
		// var gradVal G.Value
		// G.Read(grad[0], &gradVal)

		fmt.Println(g)
		fmt.Println("Here is the issue - pred Dtype:", pred.Dtype())

		vm := G.NewTapeMachine(g)
		vm.RunAll()

		if !predVal.(*tensor.Dense).Eq(out) {
			t.Errorf("expected:\n%v \nreceived:\n%v", out, pred)
		}

		fmt.Println(predVal.Shape(), in.Shape(), indices.Shape())

		vm.Reset()
		vm.Close()
	}

}
