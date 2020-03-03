package utils

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

func ReLUFloat64(n float64) float64 {
	return math.Max(0.0, n)
}

func ReLUVector(recv *mat.VecDense, v *mat.VecDense) {
	for i := 0; i < recv.Len(); i++ {
		recv.SetVec(i, ReLUFloat64(v.AtVec(i)))
	}
}
