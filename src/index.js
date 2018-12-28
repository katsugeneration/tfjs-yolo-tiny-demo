import * as tf from '@tensorflow/tfjs';
import yolo, { downloadModel } from 'tfjs-yolo-tiny';

import { Webcam } from './webcam';

let model;
const webcam = new Webcam(document.getElementById('webcam'));

(async function main() {
  try {
    model = await downloadModel();

    alert("Just a heads up! We'll ask to access your webcam so that we can " +
      "detect objects in semi-real-time. \n\nDon't worry, we aren't sending " +
      "any of your images to a remote server, all the ML is being done " +
      "locally on device, and you can check out our source code on Github.");

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

    const boxes = await yolo(inputImage, model);

    inputImage.dispose();

    const t1 = performance.now();
    console.log("YOLO inference took " + (t1 - t0) + " milliseconds.");

    console.log('tf.memory(): ', tf.memory());

    clearImages();
    boxes.forEach(box => {
      const {
        top, left, bottom, right, classProb, className,
      } = box;

      if (className === 'person') {
        drawImage("./target.png", left-64, top-64)
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
  img.style.cssText = `top: ${y + webcamMainElem.offsetTop + webcamMainElem.offsetHeight / 2}; left: ${x + webcamMainElem.offsetLeft + webcamMainElem.offsetWidth / 2}; position: absolute; z-index: 10;`;
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
