package recognizer

import (
	"bytes"
	"image"
	"github.com/jdeng/gotflite"
	"encoding/json"
	"os"
	"bufio"
)

type Model struct {
	predictor *gotflite.Predictor
	labels []string
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

