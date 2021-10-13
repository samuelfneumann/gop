package distribution

import "math/rand"

func ones(size int) []float64 {
	slice := make([]float64, size)
	for i := range slice {
		slice[i] = 1.0
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
