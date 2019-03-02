package tflite

import (
	"fmt"
	"unsafe"
)

//#cgo CXXFLAGS: -stdlib=libc++ -std=c++11 -I./tensorflow -I ./tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I ./tensorflow/tensorflow/lite/kernels -I./tensorflow/tensorflow/lite/tools/make/downloads/eigen/ -I./tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp/ -I./tensorflow/tensorflow/lite/tools/make/downloads/fft2d -I./tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -I ./tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src
//#cgo LDFLAGS: -stdlib=libc++
//#cgo CFLAGS: -I.
// #include <stdio.h>
// #include <stdlib.h>
// #include "c_api.h"
import "C"

type Interpreter struct {
	model       *C.TFL_Model
	interpreter *C.TFL_Interpreter
}

type InterpreterOptions struct {
	NumThreads int
}

func errorFromStatus(status C.TFL_Status) error {
	if status == C.kTfLiteOk {
		return nil
	}
	return fmt.Errorf("Error")
}

func newInterpreterOptions(model *C.TFL_Model, opts *InterpreterOptions) (*Interpreter, error) {
	var o *C.TFL_InterpreterOptions
	if opts != nil {
		o = C.TFL_NewInterpreterOptions()
		if opts.NumThreads > 0 {
			C.TFL_InterpreterOptionsSetNumThreads(o, C.int32_t(opts.NumThreads))
		}
		defer C.TFL_DeleteInterpreterOptions(o)
	}

	interpreter := C.TFL_NewInterpreter(model, o)
	if interpreter == nil {
		return nil, fmt.Errorf("No interpreter")
	}

	return &Interpreter{model: model, interpreter: interpreter}, nil
}

func NewInterpreter(data []byte, opts *InterpreterOptions) (*Interpreter, error) {
	model := C.TFL_NewModel(unsafe.Pointer(&data[0]), C.size_t(len(data)))
	if model == nil {
		return nil, fmt.Errorf("Invalid model")
	}

	return newInterpreterOptions(model, opts)
}

func NewInterpreterFromFile(filename string, opts *InterpreterOptions) (*Interpreter, error) {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	model := C.TFL_NewModelFromFile(cs)
	if model == nil {
		return nil, fmt.Errorf("No model")
	}

	return newInterpreterOptions(model, opts)
}

func (i *Interpreter) Release() {
	if i.interpreter != nil {
		C.TFL_DeleteInterpreter(i.interpreter)
		i.interpreter = nil
	}

	if i.model != nil {
		C.TFL_DeleteModel(i.model)
		i.model = nil
	}
}

func (i *Interpreter) AllocateTensors() error {
	if i.interpreter == nil {
		return fmt.Errorf("No interpreter")
	}

	status := C.TFL_InterpreterAllocateTensors(i.interpreter)
	return errorFromStatus(status)
}

func (i *Interpreter) Invoke() error {
	if i.interpreter == nil {
		return fmt.Errorf("No interpreter")
	}

	status := C.TFL_InterpreterInvoke(i.interpreter)
	return errorFromStatus(status)
}

func (i *Interpreter) GetInputTensorCount() (int, error) {
	if i.interpreter == nil {
		return 0, fmt.Errorf("No interpreter")
	}

	n := C.TFL_InterpreterGetInputTensorCount(i.interpreter)
	return int(uintptr(n)), nil
}

func (i *Interpreter) GetInputTensor(idx int) (*Tensor, error) {
	if i.interpreter == nil {
		return nil, fmt.Errorf("No interpreter")
	}

	tensor := C.TFL_InterpreterGetInputTensor(i.interpreter, C.int(idx))
	return NewTensor(tensor), nil
}

func (i *Interpreter) GetOutputTensorCount() (int, error) {
	if i.interpreter == nil {
		return 0, fmt.Errorf("No interpreter")
	}

	n := C.TFL_InterpreterGetOutputTensorCount(i.interpreter)
	return int(n), nil
}

func (i *Interpreter) GetOutputTensor(idx int) (*Tensor, error) {
	if i.interpreter == nil {
		return nil, fmt.Errorf("No interpreter")
	}

	tensor := C.TFL_InterpreterGetOutputTensor(i.interpreter, C.int(idx))
	return NewTensor(tensor), nil
}

func (i *Interpreter) PrintState() {
	if i.interpreter == nil {
		return
	}

	C.TFL_PrintInterpreterState(i.interpreter)
}

func (i *Interpreter) ResizeInputTensors(idx int, dims []int) error {
	if i.interpreter == nil {
		return fmt.Errorf("No interpreter")
	}

	status := C.TFL_InterpreterResizeInputTensor(i.interpreter, C.int32_t(idx), (*C.int)(unsafe.Pointer(&dims[0])), C.int32_t(len(dims)))
	return errorFromStatus(status)
}

//TODO:
func (i *Interpreter) SetErrorReporter() error {
	return fmt.Errorf("Not implemented")
}

type Tensor struct {
	tensor *C.TFL_Tensor
}

func NewTensor(tensor *C.TFL_Tensor) *Tensor {
	return &Tensor{tensor: tensor}
}

func (t *Tensor) Type() int {
	typ := C.TFL_TensorType(t.tensor)
	return int(typ)
}

func (t *Tensor) Name() string {
	return C.GoString(C.TFL_TensorName(t.tensor))
}

func (t *Tensor) Dim(i int) int {
	x := C.TFL_TensorDim(t.tensor, C.int(i))
	return int(x)
}

func (t *Tensor) NumDims() int {
	return int(C.TFL_TensorNumDims(t.tensor))
}

func (t *Tensor) Dims() []int {
	var dims []int
	for i := 0; i < t.NumDims(); i += 1 {
		dims = append(dims, t.Dim(i))
	}
	return dims
}

func (t *Tensor) Data() unsafe.Pointer {
	return unsafe.Pointer(C.TFL_TensorData(t.tensor))
}

func (t *Tensor) NumElements() int {
	dims := t.Dims()
	if len(dims) == 0 {
		return 0
	}

	n := 1
	for _, x := range dims {
		n *= x
	}

	return n
}

func (t *Tensor) ByteSize() int64 {
	return int64(C.TFL_TensorByteSize(t.tensor))
}

func (t *Tensor) ToFloats() ([]float32, error) {
	if t.Type() != C.kTfLiteFloat32 {
		return nil, fmt.Errorf("Not float")
	}

	n := t.NumElements()
	p := (*[1 << 16]float32)(t.Data())[:n] //iOS asks for a smaller one
        return p[:], nil
}

func (t *Tensor) CopyFloats(data []float32) error {
	if t.Type() != C.kTfLiteFloat32 {
		return fmt.Errorf("Not float")
	}

	status := C.TFL_TensorCopyFromBuffer(t.tensor, unsafe.Pointer(&data[0]), C.size_t(len(data))*C.size_t(unsafe.Sizeof(data[0])))
	return errorFromStatus(status)
}

//TODO:
func (t *Tensor) QuantizationParams() {
}
