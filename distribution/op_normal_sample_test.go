package distribution

import (
	"fmt"
	"testing"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestNormalSample(t *testing.T) {
	g := G.NewGraph()

	meanT := tensor.NewDense(
		tensor.Float64,
		[]int{2, 2, 2},
		tensor.WithBacking([]float64{0, 1, 2, 3, 4, 5, 6, 7}),
	)
	mean := G.NewTensor(
		g,
		meanT.Dtype(),
		meanT.Dims(),
		G.WithName("mean"),
		G.WithValue(meanT),
	)

	stdT := tensor.NewDense(
		tensor.Float64,
		[]int{2, 2, 2},
		tensor.WithBacking([]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}),
	)
	std := G.NewTensor(
		g,
		stdT.Dtype(),
		stdT.Dims(),
		G.WithName("std"),
		G.WithValue(stdT),
	)

	s, err := NormalRand(mean, std, uint64(time.Now().UnixNano()),
		1)
	if err != nil {
		t.Error(err)
	}
	var sampled G.Value
	G.Read(s, &sampled)

	vm := G.NewTapeMachine(g)
	vm.RunAll()

	fmt.Println(sampled)

	vm.Reset()
	vm.Close()
}
