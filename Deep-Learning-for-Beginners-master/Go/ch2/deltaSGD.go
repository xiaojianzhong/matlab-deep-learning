package ch2

import (
	"github.com/xiaojianzhong/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
)

func DeltaSGD(
	ws *mat.VecDense, // 3-elem vector
	xss *mat.Dense, // 4 x 3 matrix
	ds *mat.VecDense, // 4-elem vector
) {
	alpha := 0.9 // scalar

	n := 4 // scalar
	for k := 0; k < n; k++ {
		xs := xss.RowView(k) // 3-elem vector
		d := ds.AtVec(k) // scalar

		v := mat.Dot(ws, xs) // scalar
		y := utils.SigmoidFloat64(v) // scalar

		e := d - y // scalar
		delta := y * (1 - y) * e // scalar

		dW := mat.NewVecDense(3, nil) // 3-elem vector
		dW.ScaleVec(alpha * delta, xs)

		ws.AddVec(ws, dW)
	}
}
