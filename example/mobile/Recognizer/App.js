import React, { Component } from 'react';
import { useState } from 'react';
import { Text, View, Image, Button, Dimensions, NativeModules } from 'react-native';
import ImagePicker from 'react-native-image-picker';

const { RNRecognizer } = NativeModules;
const dim = Dimensions.get('window');

function Recognizer() {
  const [photo, setPhoto] = useState({});
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const recognize = async (url) => {
    setIsLoading(true);
    setResults([]);
    try {
      const res = await RNRecognizer.Recognize(url)
      const json = JSON.parse(res);
      console.log('recognized', json);
      setResults(json);
    } catch (err) {
      console.log('failed to recognize', err)
    }
    setIsLoading(false);
  };

  function selectPhoto() {
    const options = {
      title: 'Select Photo',
      storageOptions: {
        skipBackup: true,
        path: 'images',
      },
    };

    ImagePicker.showImagePicker(options, (response) => {
      if (response.didCancel) {
        console.log('User cancelled image picker');
      } else if (response.error) {
        console.log('ImagePicker Error: ', response.error);
      } else {
        const uri = response.uri;
        Image.getSize(uri, (width, height) => {
          setPhoto({ uri, width, height });
          recognize(uri);
        });
      }
    });
  }

  console.log('photo:', photo);

  let { uri, width, height } = photo;
  if (width > dim.width) {
    height = Math.round(dim.width / width * height);
    width = dim.width;
  }
  if (height > dim.height) {
    width = Math.round(dim.height / height * width);
    height = dim.height;
  }

  return (<View>
    <Button onPress={selectPhoto} title='Select Photo' />
    {uri && <Image source={{ uri: uri }} style={{ width, height }} />}
    {uri && <View>
      {isLoading && <Text>Recognizing...</Text>}
      {results && results.map((x, i) => <Text key={i}>{x}</Text>)}
    </View>}
  </View>);
}

type Props = {};
export default class App extends Component<Props> {
  render() {
    return (
      <View style={{ flex: 1, justifyContent: 'center', backgroundColor: '#F5FCFF', }} >
        <Recognizer />
      </View>
    );
  }
}
