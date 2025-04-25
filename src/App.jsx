import { useState, useRef } from 'react'
import modelFilePath from '/squeezenet1_1.onnx'
import yolov9ModelFilePath from '/yolov9-c.onnx';
import './App.css'
import * as ort from 'onnxruntime-web';
import { imagenetClasses } from './imagenet';
import yaml from 'js-yaml';
import metadataYaml from '/yolov9-c.yaml';

let yoloClassNames = [];

async function loadYoloClassNames() {
  try {
    const response = await fetch(metadataYaml);
    const yamlText = await response.text();
    const metadata = yaml.load(yamlText);
    yoloClassNames = metadata.names;
  } catch (error) {
    console.error('Error loading YOLO class names:', error);
  }
}

loadYoloClassNames();

async function runSqueezeNet(image) {
  // const response = await fetch();
  // const modelFile = await response.arrayBuffer();
  const session = await ort.InferenceSession.create(modelFilePath, { executionProviders: ['webgpu'] });
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
  console.log('YOLO runtime:', ort.env);
  const input_shape= [image.height, image.width, 3];
  const inputTensor = preprocessImage(image); // Reuse preprocessImage function
  const feed = {};
  feed[session.inputNames[0]] = inputTensor;
  const output = await session.run(feed);
  const outputTensor = output[session.outputNames[1]];
  const detections = postprocessYOLOv9Output(outputTensor, input_shape);
  console.log('Detections:', detections);
  return detections;
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
    const { x, y, w, h, confidence, className } = det;
    context.strokeRect(x, y, w, h);
    context.fillText(`${className} (${confidence.toFixed(2)})`, x, y - 5);
  });
}


function nonMaximaSuppression(detections, iouThreshold = 0.5) {
  // Sort detections by confidence in descending order
  detections.sort((a, b) => b.confidence - a.confidence);

  const selectedDetections = [];

  while (detections.length > 0) {
    const best = detections.shift(); // Take the detection with the highest confidence
    selectedDetections.push(best);

    detections = detections.filter(det => {
      const iou = calculateIoU(best, det);
      return iou < iouThreshold; // Keep only boxes with IoU below the threshold
    });
  }

  return selectedDetections;
}

function calculateIoU(boxA, boxB) {
  const x1 = Math.max(boxA.x, boxB.x);
  const y1 = Math.max(boxA.y, boxB.y);
  const x2 = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
  const y2 = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = boxA.w * boxA.h;
  const areaB = boxB.w * boxB.h;

  const union = areaA + areaB - intersection;

  return intersection / union;
}

function postprocessYOLOv9Output(output, input_shape, confidenceTH = 0.1) {
  // Assuming YOLOv9 output format: [x, y, w, h, confidence, classId]
  const detections = [];
  const ratio_x = input_shape[1] / 640;
  const ratio_y = input_shape[0] / 640;

  for (let i = 0; i < output.dims[2]; i += 1) {
    const x = output.data[i] * ratio_x;
    const y = output.data[i + output.dims[2] * 1] * ratio_y;
    const w = output.data[i + output.dims[2] * 2] * ratio_x;
    const h = output.data[i + output.dims[2] * 3] * ratio_y;

    const x1 = x - w / 2;
    const y1 = y - h / 2;

    const confidences = Array.from({ length: output.dims[1] - 4 }, (_, i1) => output.data[(i1 + 4) * output.dims[2] + i]);
    const classId = confidences.indexOf(Math.max(...confidences));
    const confidence = confidences[classId];

    if (confidence > confidenceTH) { // Filter by confidence threshold
      detections.push({ x: x1, y: y1, w, h, confidence, classId, className: yoloClassNames[classId] });
    }
  }

  // Apply Non-Maxima Suppression
  return nonMaximaSuppression(detections);
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
  const inputCanvasRef = useRef(null);
  const outputCanvasRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImageFile(file);
    if (file) {
      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.onload = () => {
        const canvas = inputCanvasRef.current;
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
      const canvas = inputCanvasRef.current;
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
      const inputCanvas = inputCanvasRef.current;
      const outputCanvas = outputCanvasRef.current;
      if (!outputCanvas) return;

      const detections = await runYOLOv9(inputCanvas);
      setDetections(detections);

      // Draw bounding boxes on the output canvas
      const context = outputCanvas.getContext('2d');
      outputCanvas.width = inputCanvas.width;
      outputCanvas.height = inputCanvas.height;
      context.drawImage(inputCanvas, 0, 0);
      drawBoundingBoxes(outputCanvas, detections);
    } else {
      alert('Please upload an image or enter an image URL first.');
    }
  };

  return (
    <>
      <div className="image-upload">
        <input type="file" accept="image/*" onChange={handleImageUpload} />
        <canvas ref={inputCanvasRef} />
        <button onClick={handleRunYOLOv9}>Run YOLOv9</button>
        <div className="detections">
          {detections.length > 0 && (
            <ul>
              {detections.map((det, index) => (
                <li key={index}>
                  Class: {det.className} ({det.classId}), Confidence: {det.confidence.toFixed(2)}, BBox: ({det.x.toFixed(2)}, {det.y.toFixed(2)}, {det.w.toFixed(2)}, {det.h.toFixed(2)})
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
      <div className="output-canvas">
        <canvas ref={outputCanvasRef} />
      </div>
    </>
  );
}

export default App
