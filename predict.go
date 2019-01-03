package gotflite

import (
	"bytes"
	"image"
	//"log"
	"github.com/disintegration/imaging"
	"github.com/jdeng/gotflite/tflite"
)

type Predictor struct {
	Name string

	interpreter             *tflite.Interpreter
	imageWidth, imageHeight int
	outputTensorIndex       int
}

func NewPredictor(modelFile string, imageWidth, imageHeight int, outputTensorIndex int) (*Predictor, error) {
	intp, err := tflite.NewInterpreterFromFile(modelFile, nil)
	if err != nil {
		return nil, err
	}

	intp.AllocateTensors()

	return &Predictor{interpreter: intp, imageWidth: imageWidth, imageHeight: imageHeight, outputTensorIndex: outputTensorIndex}, nil
}

func (p *Predictor) Relase() {
	if p.interpreter != nil {
		p.interpreter.Release()
		p.interpreter = nil
	}
}

func (p *Predictor) Extract(data []byte) (output []float32, err error) {
	err = nil

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return
	}

	img = imaging.Resize(img, p.imageWidth, p.imageHeight, imaging.Linear)
	// img = imaging.Fill(img, 224, 224, imaging.Center, imaging.Linear)

	input, err := InputFrom(img, 127.5, 127.5)
	if err != nil {
		return
	}

	//get input tensor
	intp := p.interpreter
	tin, _ := intp.GetInputTensor(0)
	// log.Printf("Input dims: %v, total: %d, type: %d\n", tin.Dims(), tin.NumElements(), tin.Type())

	err = tin.CopyFloats(input)
	if err != nil {
		return
	}

	err = intp.Invoke()
	if err != nil {
		return
	}

	tout, _ := intp.GetOutputTensor(0)
	return tout.ToFloats()
}
