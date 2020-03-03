package ch2

import (
	"github.com/azxj/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"math"
	"testing"
)

func TestSGDvsBatch(t *testing.T) {
	xss := mat.NewDense(4, 3, []float64{
		0, 0, 1,
		0, 1, 1,
		1, 0, 1,
		1, 1, 1,
	}) // 4 x 3 matrix

	ds := mat.NewVecDense(4, []float64{
		0,
		0,
		1,
		1,
	}) // 4-elem vector

	es1 := mat.NewVecDense(1000, nil) // 1000-elem vector
	es2 := mat.NewVecDense(1000, nil) // 1000-elem vector

	ws1 := mat.NewVecDense(3, utils.Apply(utils.Float64s(3), func(n float64) float64 {
		return 2 * n - 1
	})) // 3-elem vector
	ws2 := mat.VecDenseCopyOf(ws1) // 3-elem vector

	// train
	for epoch := 0; epoch < 1000; epoch++ {
		DeltaSGD(ws1, xss, ds)
		DeltaBatch(ws2, xss, ds)

		e1 := 0.0 // scalar
		e2 := 0.0 // scalar
		n := 4 // scalar
		for k := 0; k < n; k++ {
			xs := xss.RowView(k) // 3-elem vector
			d := ds.AtVec(k) // scalar

			v1 := mat.Dot(ws1, xs) // scalar
			y1 := utils.SigmoidFloat64(v1) // scalar
			e1 += math.Pow(d - y1, 2)

			v2 := mat.Dot(ws2, xs) // scalar
			y2 := utils.SigmoidFloat64(v2) // scalar
			e2 += math.Pow(d - y2, 2)
		}
		es1.SetVec(epoch, e1 / float64(n))
		es2.SetVec(epoch, e2 / float64(n))
	}

	p, err := plot.New()
	if err != nil {
		t.Error(err)
	}
	p.X.Label.Text = "Epoch"
	p.Y.Label.Text = "Average of Training Error"
	pts1 := make(plotter.XYs, 1000)
	pts2 := make(plotter.XYs, 1000)
	for i := 0; i < 1000; i++ {
		pts1[i].X = float64(i + 1)
		pts1[i].Y = es1.AtVec(i)
		pts2[i].X = float64(i + 1)
		pts2[i].Y = es2.AtVec(i)
	}
	if err = plotutil.AddLines(p, "SGD", pts1, "Batch", pts2); err != nil {
		t.Error(err)
	}
	if err = p.Save(8 * vg.Inch, 8 * vg.Inch, "result.png"); err != nil {
		t.Error(err)
	}
}

