package gotflite

import (
	"image"
)

func InputFrom(img image.Image, mean, std float32) ([]float32, error) {
	bounds := img.Bounds()
	height := bounds.Max.Y - bounds.Min.Y
	width := bounds.Max.X - bounds.Min.X

	out := make([]float32, height*width*3)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// 16bit RGBA
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			pos := (y*width + x) * 3
			out[pos] = (float32(r>>8) - mean) / std
			out[pos+1] = (float32(g>>8) - mean) / std
			out[pos+2] = (float32(b>>8) - mean) / std
		}
	}

	return out, nil
}
