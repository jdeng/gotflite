A sample mobile app built with React Native (with Hooks and Native Modules), gomobile and tensorflow lite. Only iOS is implemented but Android shouldn't be difficult.

You need to install gomobile and build the binding like below in ```ios``` directory:

```gomobile bind --target=ios github.com/jdeng/gotflite/recognizer```

Only two files are of interests:
```example/mobile/Recognizer/ios/Recognizer/RNRecognizer.m``` for the native module
```example/mobile/Recognizer/App.js``` for the react native app




