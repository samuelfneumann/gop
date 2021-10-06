package gop

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/samuelfneumann/top"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type gatherOp struct {
	axis int
	dims int // Dimensions of indices tensor (equivalently output)
}

func newGatherOp(axis, dims int) (*gatherOp, error) {
	return &gatherOp{
		axis: axis,
		dims: dims,
	}, nil
}

func (g *gatherOp) Arity() int { return 2 }

func (g *gatherOp) ReturnsPtr() bool { return false }

func (g *gatherOp) CallsExtern() bool { return false }

func (g *gatherOp) OverwritesInput() int { return -1 }

func (g *gatherOp) Hashcode() uint32 { return SimpleHash(g) }

func (g *gatherOp) WriteHash(h hash.Hash) { fmt.Fprint(h, g.String()) }

func (g *gatherOp) String() string {
	return fmt.Sprintf("Gather{axis=%v, dims=%v}()", g.axis, g.dims)
}

func (g *gatherOp) DiffWRT(inputs int) []bool {
	// Differentiable WRT input, not indices
	return []bool{true, false}
}

func (g *gatherOp) SymDiff(inputs G.Nodes, output, grad *G.Node) (G.Nodes,
	error) {
	fmt.Println("SymDiff")
	err := CheckArity(g, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("symDiff: %v", err)
	}

	diffOp := &gatherDiffOp{g}
	nodes := make(G.Nodes, 1)

	nodes[0], err = G.ApplyOp(diffOp, inputs[0], inputs[1], grad)

	return nodes, err
}

func (g *gatherOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	err := CheckArity(g, len(inputs))
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}

	shapes, err := G.DimSizersToShapes(inputs)
	if err != nil {
		return nil, fmt.Errorf("inferShape: %v", err)
	}
	return shapes[1], nil
}

func (g *gatherOp) Type() hm.Type {
	any := hm.TypeVariable('a')
	indices := G.TensorType{
		Dims: g.dims,
		Of:   tensor.Int,
	}
	return hm.NewFnType(any, indices, any)
}

func (g *gatherOp) Do(inputs ...G.Value) (G.Value, error) {
	if err := g.checkInputs(inputs...); err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}
	fmt.Println("Do")

	input := inputs[0].(tensor.Tensor)
	indices := inputs[1].(tensor.Tensor)

	return top.Gather(input, g.axis, indices)
}

func (g *gatherOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(g, len(inputs)); err != nil {
		return err
	}

	t, ok := inputs[0].(tensor.Tensor)
	if !ok {
		return fmt.Errorf("expected t to be a tensor but got %T",
			inputs[0])
	} else if t == nil {
		return fmt.Errorf("cannot gather on nil tensor")
	} else if t.Size() == 0 {
		return fmt.Errorf("cannot gather on empty tensor")
	} else if g.axis >= len(t.Shape()) {
		return fmt.Errorf("axis [%v] out of range for tensor t with shape %v",
			g.axis, t.Shape())
	}

	indices, ok := inputs[1].(tensor.Tensor)
	if !ok {
		return fmt.Errorf("expected indices to be a tensor but got %T",
			inputs[1])
	} else if indices == nil {
		return fmt.Errorf("cannot gather with nil indices")
	} else if indices.Size() == 0 {
		return fmt.Errorf("cannot gather with empty indices tensor")
	} else if g.axis >= len(indices.Shape()) {
		return fmt.Errorf("axis [%v] out of range for tensor indices with "+
			"shape %v", g.axis, indices.Shape())
	}

	return nil
}

type gatherDiffOp struct {
	op *gatherOp
}

func (g *gatherDiffOp) Arity() int { return 2 }

func (g *gatherDiffOp) ReturnsPtr() bool { return false }

func (g *gatherDiffOp) CallsExtern() bool { return false }

func (g *gatherDiffOp) OverwritesInput() int { return -1 }

func (g *gatherDiffOp) Hashcode() uint32 { return SimpleHash(g) }

func (g *gatherDiffOp) WriteHash(h hash.Hash) { fmt.Fprint(h, g.String()) }

func (g *gatherDiffOp) String() string {
	return fmt.Sprintf("GatherDiff{axis=%v, dims=%v}()", g.op.axis, g.op.dims)
}

func (g *gatherDiffOp) Type() hm.Type {
	any := hm.TypeVariable('a')
	indices := G.TensorType{
		Dims: g.op.dims,
		Of:   tensor.Int,
	}

	return hm.NewFnType(any, indices, any)
}

func (g *gatherDiffOp) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	return inputs[1].(tensor.Shape), nil
}

func (g *gatherDiffOp) Do(inputs ...G.Value) (G.Value, error) {
	if err := g.checkInputs(inputs...); err != nil {
		return nil, fmt.Errorf("do: %v", err)
	}

	input := inputs[0].(tensor.Tensor)
	indices := inputs[1].(tensor.Tensor)

	fmt.Println("Do Diff")

	return top.GatherB(input, g.op.axis, indices)
}

func (g *gatherDiffOp) checkInputs(inputs ...G.Value) error {
	if err := CheckArity(g, len(inputs)); err != nil {
		return err
	}

	t, ok := inputs[0].(tensor.Tensor)
	if !ok {
		return fmt.Errorf("expected t to be a tensor but got %T",
			inputs[0])
	} else if t == nil {
		return fmt.Errorf("cannot gather on nil tensor")
	} else if t.Size() == 0 {
		return fmt.Errorf("cannot gather on empty tensor")
	} else if g.op.axis >= len(t.Shape()) {
		return fmt.Errorf("axis [%v] out of range for tensor t with shape %v",
			g.op.axis, t.Shape())
	}

	return nil
}
