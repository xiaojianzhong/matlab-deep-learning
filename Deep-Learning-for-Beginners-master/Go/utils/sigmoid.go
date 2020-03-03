package utils

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

func SigmoidFloat64(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidVector(recv *mat.VecDense, v *mat.VecDense) {
	for i := 0; i < v.Len(); i++ {
		recv.SetVec(i, SigmoidFloat64(v.AtVec(i)))
	}
}
