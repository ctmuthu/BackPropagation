// the json file (model topology) has a reference to the bin file (model weights)
const checkpointURL =
  "https://teachablemachine.withgoogle.com/models/2Smm-0ED/model.json";
// the metatadata json file contains the text labels of your model and additional information
const metadataURL =
  "https://teachablemachine.withgoogle.com/models/2Smm-0ED/metadata.json";

const size = 300;
let webcamEl;
let model;
let totalClasses;
let myCanvas;
let ctx;
let NoOfContinuousSlouchingFrames = 0;

// A function that loads the model from the checkpoint
async function load() {
  model = await tm.posenet.load(checkpointURL, metadataURL);
  totalClasses = model.getTotalClasses();
  console.log("Number of classes, ", totalClasses);
}

async function loadWebcam() {
  webcamEl = await tm.getWebcam(size, size); // can change width and height
  webcamEl.play();
}

async function setup() {
  myCanvas = createCanvas(size, size);
  ctx = myCanvas.elt.getContext("2d");
  // Call the load function, wait until it finishes loading
  await load();
  await loadWebcam();
}

function draw() {
  predictVideo(webcamEl);
}

async function predictVideo(image) {
  if (image) {
    // Prediction #1: run input through posenet
    // predictPosenet can take in an image, video or canvas html element
    const flipHorizontal = false;
    const { pose, posenetOutput } = await model.predictPosenet(
      webcamEl,
      flipHorizontal
    );
    // Prediction 2: run input through teachable machine assification model
    const prediction = await model.predict(
      posenetOutput,
      flipHorizontal,
      totalClasses
    );

    // Show the result
    const res = select('#res'); // select <span id="res">
    res.html(prediction[0].className);
    if(prediction[0].className == "Slouching"){
	NoOfContinuousSlouchingFrames = NoOfContinuousSlouchingFrames + 1;
	if(NoOfContinuousSlouchingFrames==100){
	var audio = new Audio('audio_file.mp3');
	audio.play();
	NoOfContinuousSlouchingFrames = 0;
	}
    }
    else{
	NoOfContinuousSlouchingFrames = 0;
	}
    
    // Show the probability
    const prob = select('#prob'); // select <span id="prob">
    prob.html(prediction[0].probability.toFixed(2));

    // draw the keypoints and skeleton
    if (pose) {
      const minPartConfidence = 0.5;
      ctx.drawImage(webcamEl, 0, 0);
      tm.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
      tm.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
    }
  }
}
