package gop

import (
	"fmt"
	"hash"
	"hash/fnv"
	"math/rand"

	"gorgonia.org/tensor"
)

// ariter is anything that can return its arity
type ariter interface {
	Arity() int
}

// hashWriter is anything that can write to a hash
type hashWriter interface {
	WriteHash(hash.Hash)
}

// SimpleHash constructs the 32-bit FNV-1a hash of a Gorgonia Op.
// Taken from Gorgonia.
func SimpleHash(op hashWriter) uint32 {
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

// countOnesBefore counts the number of dimensions that have length 1
// before dimension axis
func countOnesBefore(shape tensor.Shape, axis int) int {
	count := 0
	for i := 0; i < axis; i++ {
		if shape[i] == 1 {
			count++
		}
	}
	return count
}
