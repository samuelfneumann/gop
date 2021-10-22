package distribution

import "math/rand"

func ones64(size int) []float64 {
	slice := make([]float64, size)
	for i := range slice {
		slice[i] = 1.0
	}

	return slice
}

func ones32(size int) []float32 {
	slice := make([]float32, size)
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
