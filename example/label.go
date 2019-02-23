package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/jdeng/gotflite"
	"io/ioutil"
	"log"
	"os"
)

var (
	modelFile  = flag.String("model", "model.tflite", "path to tensorflow lite model file")
	imageFile  = flag.String("image", "cat15.jpg", "path to image file")
	labelsFile = flag.String("labels", "labels.txt", "path to labels file")
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

	// load model
	pred, err := gotflite.NewPredictor(*modelFile, 224, 224, 0)
	if err != nil {
		panic(err)
	}
	defer pred.Release()

	// load image
	img, err := ioutil.ReadFile(*imageFile)
	if err != nil {
		panic(err)
	}

	out, err := pred.Run(img)
	if err != nil {
		fmt.Printf("%v\n", err)
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
