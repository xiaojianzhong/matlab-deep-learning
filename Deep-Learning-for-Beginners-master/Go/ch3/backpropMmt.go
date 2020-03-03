package ch3

import (
	"github.com/azxj/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
)

func BackpropMmt(
	wss1 *mat.Dense, // 4 x 3 matrix
	ws2 *mat.VecDense, // 4-elem vector
	xss *mat.Dense, // 4 x 3 matrix
	ds *mat.VecDense, // 4-elem vector
) {
	alpha := 0.9 // scalar
	beta := 0.9 // scalar

	mmtss1 := mat.NewDense(4, 3, nil) // 4 x 3 matrix
	mmts2 := mat.NewVecDense(4, nil) // 4-elem vector

	n := 4 // scalar
	for k := 0; k < n; k++ {
		xs := xss.RowView(k) // 3-elem vector
		d := ds.AtVec(k) // scalar

		// forward propagation
		v1 := mat.NewVecDense(4, nil) // 4-elem vector
		v1.MulVec(wss1, xs)
		y1 := mat.NewVecDense(4, nil) // 4-elem vector
		utils.SigmoidVector(y1, v1)
		v2 := mat.Dot(ws2, y1) // scalar
		y2 := utils.SigmoidFloat64(v2) // scalar

		// backward propagation
		e2 := d - y2 // scalar
		delta2 := y2 * (1 - y2) * e2 // scalar
		dW2 := mat.NewVecDense(4, nil) // 4-elem vector
		dW2.ScaleVec(alpha * delta2, y1)
		mmts2.AddScaledVec(dW2, beta, mmts2)
		ws2.AddVec(ws2, mmts2)

		e1 := mat.NewVecDense(4, nil) // 4-elem vector
		e1.ScaleVec(delta2, ws2)
		delta1 := mat.NewVecDense(4, nil) // 4-elem vector
		for i := 0; i < 4; i++ {
			delta1.SetVec(i, y1.AtVec(i) * (1-y1.AtVec(i)) * e1.AtVec(i))
		}
		dW1 := mat.NewDense(4, 3, nil) // 4 x 3 matrix
		dW1.Apply(func(i, j int, v float64) float64 {
			return alpha * delta1.AtVec(i) * xs.AtVec(j)
		}, dW1)
		mmtss1.Apply(func(i, j int, v float64) float64 {
			return dW1.At(i, j) + beta * mmtss1.At(i, j)
		}, mmtss1)
		wss1.Add(wss1, mmtss1)
	}
}
