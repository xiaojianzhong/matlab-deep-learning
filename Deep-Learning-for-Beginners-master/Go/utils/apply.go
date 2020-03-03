package utils

func Apply(ns []float64, mapper func(n float64) float64) []float64 {
	applied := make([]float64, len(ns))
	for i, n := range ns {
		applied[i] = mapper(n)
	}
	return applied
}
