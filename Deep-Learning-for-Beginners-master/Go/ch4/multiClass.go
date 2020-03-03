package ch4

import (
	"github.com/azxj/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
)

func MultiClass(
	wss1 *mat.Dense, // 50 x 25 matrix
	wss2 *mat.Dense, // 5 x 50 matrix
	xss *mat.Dense, // 5 x 25 matrix
	dss *mat.Dense, // 5 x 5 matrix
) {
	alpha := 0.9 // scalar

	n := 5 // scalar
	for k := 0; k < n; k++ {
		xs := xss.RowView(k) // 25-elem vector
		ds := dss.RowView(k) // 5-elem vector

		// forward propagation
		vs1 := mat.NewVecDense(50, nil) // 50-elem vector
		vs1.MulVec(wss1, xs)
		ys1 := mat.NewVecDense(50, nil) // 50-elem vector
		utils.SigmoidVector(ys1, vs1)

		vs2 := mat.NewVecDense(5, nil) // 5-elem vector
		vs2.MulVec(wss2, ys1)
		ys2 := mat.NewVecDense(5, nil) // 5-elem vector
		utils.Softmax(ys2, vs2)

		// backward propagation
		es2 := mat.NewVecDense(5, nil) // 5-elem vector
		es2.SubVec(ds, ys2)
		deltas2 := mat.VecDenseCopyOf(es2) // 5-elem vector
		dWs2 := mat.NewDense(5, 50, nil) // 5 x 50 matrix
		dWs2.Apply(func(i, j int, v float64) float64 {
			return alpha * deltas2.AtVec(i) * ys1.AtVec(j)
		}, dWs2)
		wss2.Add(wss2, dWs2)

		es1 := mat.NewVecDense(50, nil) // 50-elem vector
		es1.MulVec(wss2.T(), deltas2)
		deltas1 := mat.NewVecDense(50, nil) // 50-elem vector
		for i := 0; i < 50; i++ {
			deltas1.SetVec(i, ys1.AtVec(i)*(1-ys1.AtVec(i))*es1.AtVec(i))
		}
		dWs1 := mat.NewDense(50, 25, nil) // 50 x 25 matrix
		dWs1.Apply(func(i, j int, v float64) float64 {
			return alpha * deltas1.AtVec(i) * xs.AtVec(j)
		}, dWs1)
		wss1.Add(wss1, dWs1)
	}
}
