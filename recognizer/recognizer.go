package recognizer

import (
	"bufio"
	"bytes"
	"encoding/json"
	"github.com/jdeng/gotflite"
	"image"
	"io/ioutil"
	"os"
	"strings"
)

type Model struct {
	predictor *gotflite.Predictor
	labels    []string
}

func New(modelFile string, labelsFile string) (*Model, error) {
	synset, err := os.Open(labelsFile)
	if err != nil {
		return nil, err
	}
	dict := []string{}
	scanner := bufio.NewScanner(synset)
	for scanner.Scan() {
		dict = append(dict, scanner.Text())
	}

	pred, err := gotflite.NewPredictor(modelFile, 224, 224, 0)
	if err != nil {
		return nil, err
	}

	return &Model{predictor: pred, labels: dict}, nil
}

func Cleanup(reco *Model) {
	if reco.predictor != nil {
		reco.predictor.Cleanup()
	}
}

func Run(reco *Model, data []byte, n int) (string, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return "", err
	}

	out, err := reco.predictor.Run(img)
	if err != nil {
		return "", err
	}

	index := make([]int, len(out))
	gotflite.Argsort(out, index)

	var results []string
	for i := 0; i < n; i++ {
		results = append(results, reco.labels[index[i]])
	}

	bs, err := json.Marshal(results)
	return string(bs), nil
}

func RunFile(reco *Model, path string, n int) (string, error) {
	if strings.HasPrefix(path, "file://") {
		path = strings.TrimPrefix(path, "file://")
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return "", err
	}
	return Run(reco, data, n)
}
