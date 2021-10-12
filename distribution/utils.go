package distribution

func ones(size int) []float64 {
	slice := make([]float64, size)
	for i := range slice {
		slice[i] = 1.0
	}

	return slice
}
