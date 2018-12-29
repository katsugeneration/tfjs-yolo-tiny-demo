import * as tf from '@tensorflow/tfjs';
import yolo, { downloadModel } from 'tfjs-yolo-tiny';

import { Webcam } from './webcam';

let model;
const webcam = new Webcam(document.getElementById('webcam'));
const class_selector = document.getElementById('class-selector');

const class_names = [
  'person',
  'bicycle',
  'car',
  'motorbike',
  'bird',
  'cat',
  'dog',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'sports ball',
  'bottle',
  'cup',
  'fork',
  'knife',
  'spoon',
  'banana',
  'apple',
  'orange',
  'chair',
  'sofa',
  'bed',
  'diningtable',
  'laptop',
  'mouse',
  'keyboard',
  'cell phone',
  'book',
  'clock',
];

for (var c of class_names) {
  var opt = document.createElement("option");
  opt.value = c;
  opt.text = c;
  class_selector.add(opt);
}

(async function main() {
  try {
    model = await downloadModel();
    await webcam.setup();

    doneLoading();
    run();
  } catch(e) {
    console.error(e);
    showError();
  }
})();

async function run() {
  while (true) {
    const inputImage = webcam.capture();

    const t0 = performance.now();

    const boxes = await yolo(inputImage, model, {maxBoxes: 2});

    inputImage.dispose();

    const t1 = performance.now();
    console.log("YOLO inference took " + (t1 - t0) + " milliseconds.");
    console.log('tf.memory(): ', tf.memory());

    clearImages();
    boxes.forEach(box => {
      const {
        top, left, bottom, right, classProb, className,
      } = box;

      if (className === class_selector.value) {
        drawImage("./target.png", left+(right-left)/2-64, top+(bottom-top)/2-64)
      }
    });

    await tf.nextFrame();
  }
}

const webcamElem = document.getElementById('webcam-wrapper');
const webcamMainElem = document.getElementById('webcam');

function drawImage(src, x, y) {
  const img = document.createElement('img');
  img.src = src;
  img.classList.add('img-target');
  img.style.cssText = `top: ${y}; left: ${x}; position: absolute; z-index: 10;`;
  webcamElem.appendChild(img);
}

function clearImages() {
  const images = document.getElementsByClassName('img-target');
  while(images[0]) {
    images[0].parentNode.removeChild(images[0]);
  }
}

function drawRect(x, y, w, h, text = '', color = 'red') {
  const rect = document.createElement('div');
  rect.classList.add('rect');
  rect.style.cssText = `top: ${y}; left: ${x}; width: ${w}; height: ${h}; border-color: ${color}`;

  const label = document.createElement('div');
  label.classList.add('label');
  label.innerText = text;
  rect.appendChild(label);

  webcamElem.appendChild(rect);
}

function clearRects() {
  const rects = document.getElementsByClassName('rect');
  while(rects[0]) {
    rects[0].parentNode.removeChild(rects[0]);
  }
}

function doneLoading() {
  const elem = document.getElementById('loading-message');
  elem.style.display = 'none';

  const successElem = document.getElementById('success-message');
  successElem.style.display = 'block';

  const webcamElem = document.getElementById('webcam-wrapper');
  webcamElem.style.display = 'flex';
}

function showError() {
  const elem = document.getElementById('error-message');
  elem.style.display = 'block';
  doneLoading();
}
