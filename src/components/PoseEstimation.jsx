import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';

const PoseEstimation = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [behaviorHistory, setBehaviorHistory] = useState([]);

  const loadModel = async () => {
    const loadedModel = await posenet.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.5,
    });
    setModel(loadedModel);
    console.log('Pose estimation model loaded');
  };

  const detect = async () => {
    if (model == null) return;

    const video = webcamRef.current.video;
    const videoWidth = webcamRef.current.video.videoWidth;
    const videoHeight = webcamRef.current.video.videoHeight;

    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    const pose = await model.estimateSinglePose(video);

    const ctx = canvasRef.current.getContext('2d');
    drawKeypoints(pose.keypoints, 0.5, ctx);
    drawSkeleton(pose.keypoints, 0.5, ctx);

    analyzePose(pose);
  };

  const analyzePose = (pose) => {
    const newBehaviors = [];

    // Check if standing
    const leftHip = pose.keypoints.find(k => k.part === 'leftHip');
    const leftKnee = pose.keypoints.find(k => k.part === 'leftKnee');
    if (leftHip && leftKnee && Math.abs(leftHip.position.y - leftKnee.position.y) > 100) {
      newBehaviors.push('Standing');
    } else {
      newBehaviors.push('Sitting');
    }

    // Check if arms are raised
    const leftShoulder = pose.keypoints.find(k => k.part === 'leftShoulder');
    const leftWrist = pose.keypoints.find(k => k.part === 'leftWrist');
    if (leftShoulder && leftWrist && leftWrist.position.y < leftShoulder.position.y) {
      newBehaviors.push('Raising arms');
    }

    // Check head tilt
    const leftEye = pose.keypoints.find(k => k.part === 'leftEye');
    const rightEye = pose.keypoints.find(k => k.part === 'rightEye');
    if (leftEye && rightEye) {
      const headTilt = Math.abs(leftEye.position.y - rightEye.position.y);
      if (headTilt > 10) {
        newBehaviors.push('Tilting head');
      }
    }

    // Check if leaning forward
    const nose = pose.keypoints.find(k => k.part === 'nose');
    if (nose && leftShoulder && nose.position.x < leftShoulder.position.x) {
      newBehaviors.push('Leaning forward');
    }

    // Update behavior history
    updateBehaviorHistory(newBehaviors);
  };

  const updateBehaviorHistory = (newBehaviors) => {
    setBehaviorHistory(prevHistory => {
      const currentTime = new Date().toLocaleTimeString();
      const newEntries = newBehaviors.map(behavior => ({
        time: currentTime,
        behavior: behavior
      }));
      
      // Combine new behaviors with previous history, keeping only the last 10 entries
      const updatedHistory = [...newEntries, ...prevHistory].slice(0, 10);
      
      return updatedHistory;
    });
  };

  const drawKeypoints = (keypoints, minConfidence, ctx) => {
    keypoints.forEach((keypoint) => {
      if (keypoint.score >= minConfidence) {
        const { y, x } = keypoint.position;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'aqua';
        ctx.fill();
      }
    });
  };

  const drawSkeleton = (keypoints, minConfidence, ctx) => {
    const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);

    adjacentKeyPoints.forEach((keypoints) => {
      drawSegment(
        toTuple(keypoints[0].position),
        toTuple(keypoints[1].position),
        ctx
      );
    });
  };

  const toTuple = ({ y, x }) => [y, x];

  const drawSegment = ([ay, ax], [by, bx], ctx) => {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'aqua';
    ctx.stroke();
  };

  useEffect(() => {
    loadModel();
  }, []);

  useEffect(() => {
    if (model) {
      const interval = setInterval(() => {
        detect();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [model]);

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
        <h2 className="text-xl font-bold mb-2">Recent Behaviors:</h2>
        <ul className="list-disc pl-5">
          {behaviorHistory.map((entry, index) => (
            <li key={index}>{entry.time}: {entry.behavior}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default PoseEstimation;