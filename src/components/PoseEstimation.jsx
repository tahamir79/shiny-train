import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import * as blazeface from '@tensorflow-models/blazeface';

const PoseEstimation = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [poseModel, setPoseModel] = useState(null);
  const [faceModel, setFaceModel] = useState(null);
  const [summary, setSummary] = useState('Initializing...');
  const [lastBehaviors, setLastBehaviors] = useState([]);

  const loadModels = async () => {
    const loadedPoseModel = await posenet.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.5,
    });
    const loadedFaceModel = await blazeface.load();
    setPoseModel(loadedPoseModel);
    setFaceModel(loadedFaceModel);
    console.log('Pose and face detection models loaded');
  };

  const detect = async () => {
    if (poseModel == null || faceModel == null) return;

    const video = webcamRef.current.video;
    const pose = await poseModel.estimateSinglePose(video);
    const predictions = await faceModel.estimateFaces(video, false);

    analyzePose(pose);
    drawFaceBoxes(predictions);
  };

  const analyzePose = (pose) => {
    const behaviors = [];

    // Check if standing
    const leftHip = pose.keypoints.find(k => k.part === 'leftHip');
    const leftKnee = pose.keypoints.find(k => k.part === 'leftKnee');
    const rightHip = pose.keypoints.find(k => k.part === 'rightHip');
    const rightKnee = pose.keypoints.find(k => k.part === 'rightKnee');

    if (leftHip && leftKnee && rightHip && rightKnee) {
      const hipHeight = Math.abs(leftHip.position.y - rightHip.position.y);
      const kneeHeight = Math.abs(leftKnee.position.y - rightKnee.position.y);
      if (hipHeight < 50 && kneeHeight < 50) {
        behaviors.push('Standing');
      } else {
        behaviors.push('Sitting');
      }
    }

    // Check if arms are raised
    const leftShoulder = pose.keypoints.find(k => k.part === 'leftShoulder');
    const leftWrist = pose.keypoints.find(k => k.part === 'leftWrist');
    const rightShoulder = pose.keypoints.find(k => k.part === 'rightShoulder');
    const rightWrist = pose.keypoints.find(k => k.part === 'rightWrist');
    if (leftShoulder && leftWrist && rightShoulder && rightWrist) {
      if (leftWrist.position.y < leftShoulder.position.y || rightWrist.position.y < rightShoulder.position.y) {
        behaviors.push('Raising arms');
      }
    }

    // Check head tilt
    const leftEye = pose.keypoints.find(k => k.part === 'leftEye');
    const rightEye = pose.keypoints.find(k => k.part === 'rightEye');
    if (leftEye && rightEye) {
      const headTilt = Math.abs(leftEye.position.y - rightEye.position.y);
      if (headTilt > 10) {
        behaviors.push('Tilting head');
      }
    }

    // Check if leaning forward
    const nose = pose.keypoints.find(k => k.part === 'nose');
    if (nose && leftShoulder && nose.position.x < leftShoulder.position.x) {
      behaviors.push('Leaning forward');
    }

    // Check if blinking (simple heuristic based on eye position)
    if (leftEye && rightEye) {
      const eyeDistance = Math.abs(leftEye.position.y - rightEye.position.y);
      if (eyeDistance < 5) {
        behaviors.push('Blinking');
      }
    }

    // Update summary if new behaviors are detected
    if (JSON.stringify(behaviors) !== JSON.stringify(lastBehaviors)) {
      setLastBehaviors(behaviors);
      setSummary(`You are currently ${behaviors.join(', ')}.`);
    }
  };

  const drawFaceBoxes = (predictions) => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height); // Clear previous drawings

    predictions.forEach(prediction => {
      const [x1, y1] = prediction.topLeft;
      const [x2, y2] = prediction.bottomRight;

      ctx.strokeStyle = 'yellow';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
  };

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (poseModel && faceModel) {
      const interval = setInterval(() => {
        detect();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [poseModel, faceModel]);

  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        <Webcam
          ref={webcamRef}
          style={{
            width: 640,
            height: 480,
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: 640,
            height: 480,
          }}
        />
      </div>
      <div className="mt-4 p-4 bg-white rounded shadow-md w-full max-w-md">
        <h2 className="text-xl font-bold mb-2">Current Activity:</h2>
        <p>{summary}</p>
      </div>
    </div>
  );
};

export default PoseEstimation;