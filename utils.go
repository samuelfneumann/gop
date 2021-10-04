package gop

import (
	"fmt"
	"hash/fnv"
	"math/rand"

	"gorgonia.org/gorgonia"
)

// ariter is anything that can return its arity
type ariter interface {
	Arity() int
}

// SimpleHash constructs the 32-bit FNV-1a hash of a Gorgonia Op.
// Taken from Gorgonia.
func SimpleHash(op gorgonia.Op) uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func CheckArity(op ariter, inputs int) error {
	if inputs != op.Arity() && op.Arity() >= 0 {
		return fmt.Errorf("%v has an arity of %d. Got %d instead", op,
			op.Arity(), inputs)
	}
	return nil
}

// randF64 returns a random float64 slice of length size
func randF64(size int, min, max float64) []float64 {
	slice := make([]float64, size)
	for i := range slice {
		slice[i] = min + rand.Float64()*(max-min)
	}

	return slice
}

// randF32 returns a random float32 slice of length size
func randF32(size int, min, max float32) []float32 {
	slice := make([]float32, size)
	for i := range slice {
		slice[i] = min + float32(rand.Float64())*(max-min)
	}

	return slice
}

// randInt returns a random int slice of length size
func randInt(size int, min, max int) []int {
	slice := make([]int, size)
	for i := range slice {
		slice[i] = min + rand.Intn(max-min)
	}

	return slice
}
