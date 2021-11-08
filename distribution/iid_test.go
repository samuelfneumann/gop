package distribution

import (
	"fmt"
	"testing"
	"time"

	rand "golang.org/x/exp/rand"

	"github.com/samuelfneumann/gop"
	"gonum.org/v1/gonum/mat"
	mv "gonum.org/v1/gonum/stat/distmv"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestIIDProb(t *testing.T) {

	shape := []int{2, 3}
	numDists := tensor.ProdInts(shape)
	meanBacking := make([]float64, numDists)
	stdBacking := make([]float64, numDists)
	for r := 0; r < numDists; r++ {
		mean := 0.
		stddev := 1.
		meanBacking[r] = mean
		stdBacking[r] = stddev
	}

	stdDense := mat.NewDiagDense(len(stdBacking), stdBacking)
	src := rand.NewSource(uint64(time.Now().UnixNano()))
	targetDist, ok := mv.NewNormal(meanBacking, stdDense, src)
	if !ok {
		t.Error("could not construct target normal")
	}

	batchSize := 5
	dataSlice := make([]float64, numDists*batchSize)
	dataShape := append([]int{batchSize}, shape...)
	for i := range dataSlice {
		dataSlice[i] = 0
	}

	g := G.NewGraph()

	meanT := tensor.NewDense(
		tensor.Float64,
		shape,
		tensor.WithBacking(meanBacking),
	)
	stdT := tensor.NewDense(
		tensor.Float64,
		shape,
		tensor.WithBacking(stdBacking),
	)
	dataT := tensor.NewDense(
		tensor.Float64,
		dataShape,
		tensor.WithBacking(dataSlice),
	)

	mean := G.NewTensor(
		g,
		tensor.Float64,
		meanT.Dims(),
		G.WithValue(meanT),
		G.WithName(gop.Unique("mean")),
	)

	std := G.NewTensor(
		g,
		tensor.Float64,
		meanT.Dims(),
		G.WithValue(stdT),
		G.WithName(gop.Unique("std")),
	)

	data := G.NewTensor(
		g,
		tensor.Float64,
		dataT.Dims(),
		G.WithShape(dataShape...),
		G.WithValue(dataT),
		G.WithName(gop.Unique("input")),
	)

	n, err := NewNormal(mean, std, 1)
	if err != nil {
		t.Error(err)
	}

	i := NewIID(n, 1)

	prob, err := i.Prob(data)
	if err != nil {
		t.Error(err)
	}
	var computedProb G.Value
	G.Read(prob, &computedProb)

	vm := G.NewTapeMachine(g)

	vm.RunAll()

	fmt.Println(computedProb)
	fmt.Println(targetDist.Prob(dataSlice[:numDists]))

	vm.Close()
}
