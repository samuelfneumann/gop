package op

import (
	"fmt"
	"hash/fnv"

	"gorgonia.org/gorgonia"
)

// SimpleHash constructs the 32-bit FNV-1a hash of a Gorgonia Op.
// Taken from Gorgonia.
func SimpleHash(op gorgonia.Op) uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func CheckArity(op gorgonia.Op, inputs int) error {
	if inputs != op.Arity() && op.Arity() >= 0 {
		return fmt.Errorf("%v has an arity of %d. Got %d instead", op,
			op.Arity(), inputs)
	}
	return nil
}
