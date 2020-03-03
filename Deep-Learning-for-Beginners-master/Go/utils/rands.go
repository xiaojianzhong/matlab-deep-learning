package utils

import (
	"math/rand"
)

func Float64s(size int) []float64 {
	ns := make([]float64, size)
	for i := range ns {
		ns[i] = rand.Float64()
	}
	return ns
}
