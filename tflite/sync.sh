export TFLITE=$TFROOT/tensorflow/tensorflow/lite
mkdir -p t/tensorflow/lite
cp -a $TFLITE/*.h $TFLITE/*.cc t/tensorflow/lite/
for d in "c" "core" "delegates" "experimental" "kernels" "lib_package" "nnapi" "profiling" "schema"; do
cp -r $TFLITE/$d t/tensorflow/lite/
done

