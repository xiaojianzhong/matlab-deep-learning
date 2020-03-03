package utils

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

func Softmax(recv *mat.VecDense, v *mat.VecDense) {
	ex := mat.NewVecDense(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		ex.SetVec(i, math.Exp(v.AtVec(i)))
	}
	recv.ScaleVec(1.0 / mat.Sum(ex.T()), ex)
}
