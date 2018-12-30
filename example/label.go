package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/disintegration/imaging"
	"github.com/jdeng/gotflite"
	"github.com/jdeng/gotflite/tflite"
	"image"
	"image/jpeg"
	"log"
	"os"
)

var (
	modelFile  = flag.String("model", "model.tflite", "path to tensorflow lite model file")
	imageFile  = flag.String("image", "cat15.jpg", "path to image file")
	labelsFile = flag.String("labels", "labels.txt", "path to labels file")
	saveImage  = flag.String("saveimage", "", "path to save resized image")
)

func main() {
	flag.Parse()

	// load labels
	synset, err := os.Open(*labelsFile)
	if err != nil {
		panic(err)
	}
	dict := []string{}
	scanner := bufio.NewScanner(synset)
	for scanner.Scan() {
		dict = append(dict, scanner.Text())
	}

	// load image
	reader, err := os.Open(*imageFile)
	if err != nil {
		panic(err)
	}

	img, _, _ := image.Decode(reader)
	img = imaging.Fill(img, 224, 224, imaging.Center, imaging.Lanczos)

	if *saveImage != "" {
		f, _ := os.Create(*saveImage)
		defer f.Close()
		jpeg.Encode(f, img, nil)
	}

	input, err := gotflite.InputFrom(img, 127.5, 127.5)
	if err != nil {
		panic(err)
	}
	log.Printf("Image data loaded from %s\n", *imageFile)

	// load model
	intp, err := tflite.NewInterpreterFromFile(*modelFile)
	if err != nil {
		panic(err)
	}
	defer intp.Release()
	//	intp.PrintState()
	if err := intp.AllocateTensors(); err != nil {
		panic(err)
	}

	log.Printf("Interpreter created from %s\n", *modelFile)

	//get input tensor
	tin, _ := intp.GetInputTensor(0)
	log.Printf("Input dims: %v, total: %d, type: %d\n", tin.Dims(), tin.NumElements(), tin.Type())

	if err := tin.CopyFloats(input); err != nil {
		panic(err)
	}

	if err := intp.Invoke(); err != nil {
		panic(err)
	}

	tout, _ := intp.GetOutputTensor(0)
	log.Printf("Output dims: %v, total: %d, type: %d\n", tout.Dims(), tout.NumElements(), tout.Type())

	out, err := tout.ToFloats()
	if err != nil {
		panic(err)
	}

	index := make([]int, len(out))
	gotflite.Argsort(out, index)

	log.Println("Top results:")
	for i := 0; i < 10; i++ {
		fmt.Printf(" %d, %f, %s\n", index[i], out[i], dict[index[i]])
	}
	fmt.Println("")
}
