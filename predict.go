package gotflite

import (
	"bytes"
	"fmt"
	"github.com/disintegration/imaging"
	"github.com/jdeng/gotflite/tflite"
	"image"
)

type Predictor struct {
	Name string

	interpreter             *tflite.Interpreter
	imageWidth, imageHeight int
	ImageProcessor          func(img image.Image, width, height int) image.Image
	outputTensorIndex       int
}

func NewPredictor(modelFile string, imageWidth, imageHeight int, outputTensorIndex int) (*Predictor, error) {
	intp, err := tflite.NewInterpreterFromFile(modelFile, nil)
	if err != nil {
		return nil, err
	}

	if err = intp.AllocateTensors(); err != nil {
		return nil, err
	}

	return &Predictor{interpreter: intp, imageWidth: imageWidth, imageHeight: imageHeight, outputTensorIndex: outputTensorIndex}, nil
}

func (p *Predictor) Release() {
	if p.interpreter != nil {
		p.interpreter.Release()
		p.interpreter = nil
	}
}

func (p *Predictor) Run(data []byte) (output []float32, err error) {
	err = nil

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		fmt.Printf("Failed to decode %v\n", err)
		return
	}

	if p.ImageProcessor == nil {
		img = imaging.Resize(img, p.imageWidth, p.imageHeight, imaging.Linear)
	} else {
		img = p.ImageProcessor(img, p.imageWidth, p.imageHeight)
	}
	// img = imaging.Fill(img, p.imageWidth, p.imageHeight, imaging.Center, imaging.Linear)

	input, err := InputFrom(img, 127.5, 127.5)
	if err != nil {
		fmt.Printf("Failed to load image: %v\n", err)
		return
	}

	//get input tensor
	intp := p.interpreter
	tin, err := intp.GetInputTensor(0)
	fmt.Printf("Input dims: %v, total: %d, type: %d, img: %d\n", tin.Dims(), tin.NumElements(), tin.Type(), len(input))
	if err != nil {
		fmt.Printf("Failed to get input tensor: %v\n", err)
		return
	}

	err = tin.CopyFloats(input)
	if err != nil {
		fmt.Printf("Failed to copy input: %v\n", err)
		return
	}

	err = intp.Invoke()
	if err != nil {
		fmt.Printf("Failed to invoke: %v\n", err)
		return
	}

	tout, _ := intp.GetOutputTensor(p.outputTensorIndex)
	return tout.ToFloats()
}
