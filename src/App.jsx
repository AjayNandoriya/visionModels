import { useState, useRef } from 'react'
import modelFilePath from '/squeezenet1_1.onnx'
import yolov9ModelFilePath from '/yolov9-c.onnx';
import './App.css'
import * as ort from 'onnxruntime-web';
import { imagenetClasses } from './imagenet';

async function runSqueezeNet(image) {
  // const response = await fetch();
  // const modelFile = await response.arrayBuffer();
  const session = await ort.InferenceSession.create(modelFilePath, { executionProviders: ['cpu'] });
  const inputTensor = preprocessImage(image);
  const feed = {};
  feed[session.inputNames[0]] = inputTensor;
  const output = await session.run(feed);
  const outputTensor = output[session.outputNames[0]];
  const classInfo = postprocessOutput(outputTensor);
  console.log('Predicted class:', classInfo.name, 'with probability:', classInfo.prob);
  return classInfo;
}

// write a function to convert image to canvas
// and then to tensor
// using onnxruntime-web

function preprocessImage(image) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  // Set canvas dimensions to match the image
  canvas.width = image.width;
  canvas.height = image.height;

  // Draw the image onto the canvas
  context.drawImage(image, 0, 0, image.width, image.height);

  // Get image data from the canvas
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

  // Convert image data to a Float32Array and normalize pixel values
  const floatArray = new Float32Array(imageData.data.length / 4 * 3); // RGB only
  for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
    floatArray[j] = (imageData.data[i] / 255.0 - 0.485)/0.229;     // R
    floatArray[j + 1] = (imageData.data[i + 1] / 255.0 - 0.456)/0.224; // G
    floatArray[j + 2] = (imageData.data[i + 2] / 255.0 - 0.406)/0.225; // B
  }

  // resize image to 224x224
  // const H = 224;
  // const W = 224;
  const H = 640;
  const W = 640;
  const resizedTensor = new ort.Tensor('float32', new Float32Array(3 * H * W), [1, 3, H, W]);
  const resizedData = resizedTensor.data;
  for (let i = 0; i < 3; i++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const srcX = Math.floor(x * (canvas.width / W));
        const srcY = Math.floor(y * (canvas.height / H));
        const srcIndex = (srcY * canvas.width + srcX) * 3 + i;
        const destIndex = ((i * H + y) * W + x);
        resizedData[destIndex] = floatArray[srcIndex];
      }
    }
  }
  return resizedTensor;
}

function postprocessOutput(output) {
  // Apply softmax to the output tensor
  const expOutput = output.data.map(Math.exp);
  const sumExp = expOutput.reduce((a, b) => a + b, 0);
  const softmaxOutput = expOutput.map(x => x / sumExp);
  // Find the index of the maximum value in the softmax output
  const maxIndex = softmaxOutput.indexOf(Math.max(...softmaxOutput));
  return {name: imagenetClasses[maxIndex][1],
          prob: softmaxOutput[maxIndex]};
}

async function runYOLOv9(image) {
  const session = await ort.InferenceSession.create(yolov9ModelFilePath, { executionProviders: ['cpu'] });
  const inputTensor = preprocessImage(image); // Reuse preprocessImage function
  const feed = {};
  feed[session.inputNames[0]] = inputTensor;
  const output = await session.run(feed);
  const outputTensor = output[session.outputNames[0]];
  const detections = postprocessYOLOv9Output(outputTensor);
  console.log('Detections:', detections);
  return detections;
}

function postprocessYOLOv9Output(output) {
  // Assuming YOLOv9 output format: [x, y, w, h, confidence, classId]
  const detections = [];
  for (let i = 0; i < output.data.length; i += 6) {
    const [x, y, w, h, confidence, classId] = output.data.slice(i, i + 6);
    if (confidence > 0.5) { // Filter by confidence threshold
      detections.push({ x, y, w, h, confidence, classId });
    }
  }
  return detections;
}

function App() {
  const [count, setCount] = useState(0)
  const [imageFile, setImageFile] = useState(null);
  const [classInfo, setClassInfo] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const canvasRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImageFile(file);
    if (file) {
      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const context = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        context.drawImage(img, 0, 0);
      };
      img.src = URL.createObjectURL(file);
    }
  };

  const handleLoadImageFromUrl = () => {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const context = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      context.drawImage(img, 0, 0);
    };
    img.src = imageUrl;
  };

  const handleRunSqueezeNet = async () => {
    if (imageFile || imageUrl) {
      const canvas = canvasRef.current;
      const classInfo = await runSqueezeNet(canvas);
      setClassInfo(classInfo);
    } else {
      alert('Please upload an image or enter an image URL first.');
    }
  };

  return (
    <>
      <div className="image-upload">
        <input type="file" accept="image/*" onChange={handleImageUpload} />
        <canvas ref={canvasRef} />
        <button onClick={handleRunSqueezeNet}>Run SqueezeNet</button>
        <div className="class-info">
          {classInfo && (
            <p>Predicted class: {classInfo.name} with probability: {classInfo.prob.toFixed(2)}</p>
          )}
        </div>
      </div>
      <div className="url-upload">
        <input
          type="text"
          placeholder="Enter image URL"
          onChange={(e) => setImageUrl(e.target.value)}
        />
        <button onClick={handleLoadImageFromUrl}>Load Image</button>
      </div>
      <YOLOv9App />
    </>
  )
}

function YOLOv9App() {
  const [imageFile, setImageFile] = useState(null);
  const [detections, setDetections] = useState([]);
  const [imageUrl, setImageUrl] = useState('');
  const canvasRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImageFile(file);
    if (file) {
      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const context = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        context.drawImage(img, 0, 0);
      };
      img.src = URL.createObjectURL(file);
    }
  };

  const handleLoadImageFromUrl = () => {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const context = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      context.drawImage(img, 0, 0);
    };
    img.src = imageUrl;
  };

  const handleRunYOLOv9 = async () => {
    if (imageFile || imageUrl) {
      const canvas = canvasRef.current;
      const detections = await runYOLOv9(canvas);
      setDetections(detections);
      drawBoundingBoxes(canvas, detections);
    } else {
      alert('Please upload an image or enter an image URL first.');
    }
  };

  return (
    <>
      <div className="image-upload">
        <input type="file" accept="image/*" onChange={handleImageUpload} />
        <canvas ref={canvasRef} />
        <button onClick={handleRunYOLOv9}>Run YOLOv9</button>
        <div className="detections">
          {detections.length > 0 && (
            <ul>
              {detections.map((det, index) => (
                <li key={index}>
                  Class: {det.classId}, Confidence: {det.confidence.toFixed(2)}, BBox: ({det.x.toFixed(2)}, {det.y.toFixed(2)}, {det.w.toFixed(2)}, {det.h.toFixed(2)})
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="url-upload">
        <input
          type="text"
          placeholder="Enter image URL"
          onChange={(e) => setImageUrl(e.target.value)}
        />
        <button onClick={handleLoadImageFromUrl}>Load Image</button>
      </div>
    </>
  );
}

function drawBoundingBoxes(canvas, detections) {
  const context = canvas.getContext('2d');
  context.strokeStyle = 'red';
  context.lineWidth = 2;
  context.font = '16px Arial';
  context.fillStyle = 'red';

  // Sort detections by confidence in descending order and take the top 5
  const topDetections = detections.sort((a, b) => b.confidence - a.confidence).slice(0, 5);

  topDetections.forEach(det => {
    const { x, y, w, h, confidence, classId } = det;
    const ratio_x = canvas.width / 640;
    const ratio_y = canvas.height / 640;
    context.strokeRect(x/ratio_x, y/ratio_y, w/ratio_x, h/ratio_y);
    context.fillText(`Class: ${classId}, Conf: ${confidence.toFixed(2)}`, x, y - 5);
  });
}

export default App
