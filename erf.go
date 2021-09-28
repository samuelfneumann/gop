package op

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
)

// Erf is the error function
type Erf struct{}

func (e Erf) Arity() int {
	return 1
}

func (e Erf) Type() hm.Type {
	return hm.TypeConst("erf")
}

func (e Erf) Do(node ...Value) (Value, error) {
	if len(node) != 1 {
		return nil, fmt.Errorf("do: expected 1 value, got %v", len(node))
	}

	if node.Dtype() != tensor.Float64 || node.Dtype() != tensor.Float32 {
		return nil, fmt.Errorf("do: expected float64 or float32, got %v", node.Dtype())
	}

	for i := 0; i < prod(node.Shape()...); i++ {
		node.Set(i, math.erf(node.Get(i)))
	}

	return retVal, nil
}

func (e Erf) ReturnsPtr() bool {
	return true
}

func (e Erf) CallsExtern() bool {
	return false
}

func (e Erf) OverwritesInput() int {
	return 0
}

func (e Erf) String() string {
	return "erf"
}

// ErfInv is the inverse error function
type ErfInv struct{}

func prod(ints ...int) int {
	total := 1
	for _, i := range ints {
		total *= i
	}
	return total
}
