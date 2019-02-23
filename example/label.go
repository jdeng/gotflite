package main

import (
	"flag"
	"fmt"
	"github.com/jdeng/gotflite/recognizer"
	"io/ioutil"
)

var (
	modelFile  = flag.String("model", "model.tflite", "path to tensorflow lite model file")
	imageFile  = flag.String("image", "cat15.jpg", "path to image file")
	labelsFile = flag.String("labels", "labels.txt", "path to labels file")
)

func main() {
	flag.Parse()

	model, err := recognizer.New(*modelFile, *labelsFile)
	if err != nil {
		panic(err)
	}

	img, err := ioutil.ReadFile(*imageFile)
	if err != nil {
		panic(err)
	}

	// run model
	n := 10
	results, err := recognizer.Run(model, img, n)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Top %d results: %v\n", n, results)
}
