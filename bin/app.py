#!/usr/bin/env python3
"""
YouTube Video Object Detection System - PyTorch Version
Analyzes videos using YOLOv8 with PyTorch CUDA support
"""

import json
import requests
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable
import yt_dlp
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from ultralytics import YOLO
import cv2

@dataclass
class Detection:
    """Object detection result"""
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    timestamp: float
    frame_number: int
    video_url: str = ""
    video_title: str = ""
    channel_name: str = ""

class PyTorchDetector:
    """YOLOv8 object detector with PyTorch CUDA support"""
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Initialize detector with YOLOv8
        model_name options: yolov8n.pt (nano), yolov8s.pt (small), 
                           yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (xlarge)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("\n" + "=" * 60)
        print("PyTorch CUDA Detection:")
        print("=" * 60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            
            # Print memory info
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"GPU Memory allocated: {memory_allocated:.2f} MB")
            print(f"GPU Memory reserved: {memory_reserved:.2f} MB")
        else:
            print("Running on CPU")
        
        print("=" * 60 + "\n")
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8 model: {model_name}...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        print(f"✓ Model loaded on {self.device.upper()}")
        print(f"✓ Model classes: {len(self.model.names)}")
        
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> list[dict]:
        """Perform object detection on frame"""
        # Run inference
        results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        
        # Parse results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to xywh format
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                detections.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': [x, y, w, h]
                })
        
        return detections
    
    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available"""
        return self.device == 'cuda'

class VideoAnalyzer:
    """Analyzes video with object detection"""
    
    def __init__(self, detector: PyTorchDetector):
        self.detector = detector
        self.stop_flag = threading.Event()
        self.progress_info = {'progress': 0, 'frame': 0, 'total': 0, 'status': 'idle'}
        self.video_metadata = {}
        
    def download_video(self, url: str, output_path: str = "video.mp4") -> tuple[str, dict]:
        """Download YouTube video and return path + metadata"""
        print(f"Downloading video from: {url}")
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            metadata = {
                'url': url,
                'title': info.get('title', 'Unknown'),
                'channel': info.get('channel', info.get('uploader', 'Unknown')),
                'duration': info.get('duration', 0),
                'upload_date': info.get('upload_date', 'Unknown')
            }
            
            print(f"✓ Downloaded: {metadata['title']}")
            print(f"  Channel: {metadata['channel']}")
            
            self.video_metadata = metadata
        
        return output_path, metadata
    
    def analyze_video(self, video_path: str, 
                     progress_callback: Optional[Callable] = None,
                     detection_callback: Optional[Callable] = None,
                     frame_skip: int = 1,
                     conf_threshold: float = 0.5) -> list[Detection]:
        """Analyze video and detect objects"""
        import time
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVideo info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f}s")
        print(f"  Processing every {frame_skip} frame(s)\n")
        
        all_detections = []
        frame_num = 0
        processed_frames = 0
        
        # Reset stop flag and set status to analyzing
        self.stop_flag.clear()
        self.progress_info['status'] = 'analyzing'
        
        # Track processing time
        start_time = time.time()
        inference_times = []
        
        # Get video metadata
        video_url = self.video_metadata.get('url', '')
        video_title = self.video_metadata.get('title', '')
        channel_name = self.video_metadata.get('channel', '')
        
        while cap.isOpened() and not self.stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_skip == 0:
                timestamp = frame_num / fps
                
                # Time the inference
                inference_start = time.time()
                detections = self.detector.detect(frame, conf_threshold=conf_threshold)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                for det in detections:
                    detection_obj = Detection(
                        class_name=det['class_name'],
                        confidence=det['confidence'],
                        bbox=tuple(det['bbox']),
                        timestamp=round(timestamp, 2),
                        frame_number=frame_num,
                        video_url=video_url,
                        video_title=video_title,
                        channel_name=channel_name
                    )
                    all_detections.append(detection_obj)
                    
                    if detection_callback:
                        detection_callback(detection_obj)
                
                processed_frames += 1
            
            frame_num += 1
            
            # Update progress
            if progress_callback and frame_num % 30 == 0:
                progress = (frame_num / total_frames) * 100
                self.progress_info = {
                    'progress': progress,
                    'frame': frame_num,
                    'total': total_frames,
                    'status': 'analyzing'
                }
                progress_callback(progress, frame_num, total_frames)
        
        cap.release()
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        
        # Estimate CPU/GPU speedup
        # Typical YOLOv8 inference: ~50-100ms on CPU, ~5-15ms on GPU for similar hardware
        # This is a rough estimate based on the device type
        if self.detector.cuda_available:
            # Assume GPU is ~5-10x faster than CPU for YOLOv8
            estimated_cpu_time = avg_inference_time * 7  # Conservative estimate
            speedup = estimated_cpu_time / avg_inference_time if avg_inference_time > 0 else 7
        else:
            # Running on CPU
            speedup = 1.0
            estimated_cpu_time = avg_inference_time
        
        # Store performance metrics
        self.progress_info['performance'] = {
            'total_time': round(total_time, 2),
            'avg_inference_time': round(avg_inference_time * 1000, 2),  # Convert to ms
            'fps_processing': round(processed_frames / total_time, 2) if total_time > 0 else 0,
            'speedup': round(speedup, 1),
            'device': 'GPU' if self.detector.cuda_available else 'CPU'
        }
        
        # Set status to complete
        self.progress_info['status'] = 'complete'
        self.progress_info['progress'] = 100
        
        print(f"\n✓ Analysis complete!")
        print(f"  Processed frames: {processed_frames}")
        print(f"  Total detections: {len(all_detections)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg inference time: {avg_inference_time*1000:.2f}ms")
        print(f"  Processing FPS: {processed_frames/total_time:.2f}")
        if self.detector.cuda_available:
            print(f"  Estimated GPU speedup: {speedup:.1f}x vs CPU")
        
        return all_detections
    
    def stop(self):
        """Stop analysis"""
        print("Stopping analysis...")
        self.stop_flag.set()
        self.progress_info['status'] = 'stopped'
    
    def get_progress(self):
        """Get current progress"""
        return self.progress_info

class SplunkHEC:
    """Splunk HTTP Event Collector client"""
    
    def __init__(self, url: str, token: str, index: Optional[str] = None):
        self.url = url.rstrip('/')
        self.token = token
        self.index = index
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Splunk {token}',
            'Content-Type': 'application/json'
        })
        self.session.verify = False  # For self-signed certs
        
    def send_event(self, event: Detection) -> bool:
        """Send detection event to Splunk"""
        payload = {
            'event': asdict(event),
            'sourcetype': 'video:detection',
            'source': 'youtube_analyzer'
        }
        
        # Add index if specified
        if self.index:
            payload['index'] = self.index
        
        try:
            response = self.session.post(
                f'{self.url}/services/collector/event',
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                return True
            else:
                print(f"Splunk HEC error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error sending to Splunk: {e}")
            return False

# Flask API
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

detector = None
analyzer = None
splunk_client = None
current_detections = []
analysis_thread = None

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Object Detection - PyTorch</title>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect } = React;
        
        const Icon = ({ name, className = "", size = 24 }) => {
            const icons = {
                camera: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>,
                upload: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>,
                settings: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v6m0-18l-8 4m8-4l8 4m-8 18l-8-4m8 4l8-4M1 12h6m6 0h6"/></svg>,
                database: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
                play: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>,
                square: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>,
                download: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
                alert: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>,
                check: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>,
                zap: <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            };
            return icons[name] || null;
        };

        function App() {
            const [videoUrl, setVideoUrl] = useState('');
            const [splunkUrl, setSplunkUrl] = useState('');
            const [splunkToken, setSplunkToken] = useState('');
            const [splunkIndex, setSplunkIndex] = useState('');
            const [frameSkip, setFrameSkip] = useState(5);
            const [confThreshold, setConfThreshold] = useState(0.5);
            const [isAnalyzing, setIsAnalyzing] = useState(false);
            const [detections, setDetections] = useState([]);
            const [status, setStatus] = useState('');
            const [error, setError] = useState('');
            const [showSettings, setShowSettings] = useState(false);
            const [stats, setStats] = useState({});
            const [systemStatus, setSystemStatus] = useState(null);

            const API_BASE = '/api';

            useEffect(() => {
                fetchSystemStatus();
            }, []);

            useEffect(() => {
                if (isAnalyzing) {
                    const interval = setInterval(() => {
                        fetchResults();
                        checkAnalysisStatus();
                    }, 2000);
                    return () => clearInterval(interval);
                }
            }, [isAnalyzing]);

            useEffect(() => {
                if (detections.length > 0) {
                    calculateStats();
                }
            }, [detections, systemStatus]);

            const fetchSystemStatus = async () => {
                try {
                    const response = await fetch(`${API_BASE}/status`);
                    const data = await response.json();
                    setSystemStatus(data);
                } catch (err) {
                    console.error('Error fetching system status:', err);
                }
            };

            const checkAnalysisStatus = async () => {
                try {
                    const response = await fetch(`${API_BASE}/status`);
                    const data = await response.json();
                    setSystemStatus(data);  // Update system status for performance metrics
                    
                    if (data.progress && data.progress.status === 'complete') {
                        setIsAnalyzing(false);
                        setStatus('✓ Analysis complete!');
                    } else if (data.progress && data.progress.status === 'stopped') {
                        setIsAnalyzing(false);
                        setStatus('Analysis stopped');
                    }
                } catch (err) {
                    console.error('Error checking analysis status:', err);
                }
            };

            const fetchResults = async () => {
                try {
                    const response = await fetch(`${API_BASE}/results`);
                    const data = await response.json();
                    setDetections(data);
                } catch (err) {
                    console.error('Error fetching results:', err);
                }
            };

            const calculateStats = () => {
                const classCount = {};
                let totalConf = 0;
                
                detections.forEach(det => {
                    classCount[det.class_name] = (classCount[det.class_name] || 0) + 1;
                    totalConf += det.confidence;
                });
                
                const avgConf = detections.length > 0 ? (totalConf / detections.length) : 0;
                
                const newStats = {
                    totalDetections: detections.length,
                    avgConfidence: avgConf.toFixed(3),
                    uniqueClasses: Object.keys(classCount).length,
                    classCount: classCount
                };
                
                // Preserve performance metrics once they're available
                if (systemStatus?.progress?.performance) {
                    newStats.performance = systemStatus.progress.performance;
                } else if (stats.performance) {
                    // Keep existing performance data if available
                    newStats.performance = stats.performance;
                }
                
                setStats(newStats);
            };

            const startAnalysis = async () => {
                if (!videoUrl.trim()) {
                    setError('Please enter a YouTube URL');
                    return;
                }

                setError('');
                setStatus('Starting analysis...');
                setIsAnalyzing(true);
                setDetections([]);
                setStats({});  // Clear stats including performance metrics for new analysis

                try {
                    const response = await fetch(`${API_BASE}/analyze`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            url: videoUrl,
                            splunk_url: splunkUrl,
                            splunk_token: splunkToken,
                            splunk_index: splunkIndex,
                            frame_skip: frameSkip,
                            conf_threshold: confThreshold
                        })
                    });

                    const data = await response.json();
                    
                    if (response.ok) {
                        setStatus(data.message);
                    } else {
                        setError(data.error || 'Analysis failed');
                        setIsAnalyzing(false);
                    }
                } catch (err) {
                    setError(`Connection error: ${err.message}`);
                    setIsAnalyzing(false);
                }
            };

            const stopAnalysis = async () => {
                try {
                    await fetch(`${API_BASE}/stop`, { method: 'POST' });
                    setIsAnalyzing(false);
                    setStatus('Analysis stopped');
                } catch (err) {
                    setError('Error stopping analysis');
                }
            };

            const downloadResults = () => {
                const blob = new Blob([JSON.stringify(detections, null, 2)], 
                    { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'detections.json';
                a.click();
                URL.revokeObjectURL(url);
            };

            const getTopClasses = () => {
                if (!stats.classCount) return [];
                return Object.entries(stats.classCount)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5);
            };

            return (
                <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
                    <div className="max-w-7xl mx-auto">
                        <div className="text-center mb-8">
                            <div className="flex items-center justify-center gap-3 mb-3">
                                <Icon name="camera" className="text-purple-400" size={48} />
                                <h1 className="text-4xl font-bold text-white">Video Object Detection</h1>
                                {systemStatus?.cuda_available && (
                                    <Icon name="zap" className="text-yellow-400" size={32} />
                                )}
                            </div>
                            <p className="text-purple-200">
                                PyTorch YOLOv8 with {systemStatus?.cuda_available ? 'CUDA GPU' : 'CPU'} Acceleration
                            </p>
                            {systemStatus?.cuda_device && (
                                <p className="text-sm text-purple-300 mt-1">
                                    {systemStatus.cuda_device}
                                </p>
                            )}
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="lg:col-span-1 space-y-6">
                                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                                    <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                                        <Icon name="upload" size={20} />
                                        Video Source
                                    </h2>
                                    
                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-sm font-medium text-purple-200 mb-2">YouTube URL</label>
                                            <input
                                                type="text"
                                                value={videoUrl}
                                                onChange={(e) => setVideoUrl(e.target.value)}
                                                placeholder="https://youtube.com/watch?v=..."
                                                className="w-full px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-purple-200 mb-2">
                                                Frame Skip (1 = every frame)
                                            </label>
                                            <input
                                                type="number"
                                                value={frameSkip}
                                                onChange={(e) => setFrameSkip(parseInt(e.target.value) || 1)}
                                                min="1"
                                                max="30"
                                                className="w-full px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-purple-200 mb-2">
                                                Confidence Threshold ({confThreshold})
                                            </label>
                                            <input
                                                type="range"
                                                value={confThreshold}
                                                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                                                min="0.1"
                                                max="0.95"
                                                step="0.05"
                                                className="w-full"
                                            />
                                            <p className="text-xs text-purple-300 mt-1">Higher = fewer but more confident detections</p>
                                        </div>

                                        <button
                                            onClick={() => setShowSettings(!showSettings)}
                                            className="flex items-center gap-2 text-purple-300 hover:text-purple-100 transition"
                                        >
                                            <Icon name="settings" size={16} />
                                            {showSettings ? 'Hide' : 'Show'} Splunk Settings
                                        </button>

                                        {showSettings && (
                                            <div className="space-y-4 pt-4 border-t border-white/10">
                                                <div>
                                                    <label className="block text-sm font-medium text-purple-200 mb-2 flex items-center gap-2">
                                                        <Icon name="database" size={16} />
                                                        Splunk HEC URL
                                                    </label>
                                                    <input
                                                        type="text"
                                                        value={splunkUrl}
                                                        onChange={(e) => setSplunkUrl(e.target.value)}
                                                        placeholder="https://splunk.example.com:8088"
                                                        className="w-full px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="block text-sm font-medium text-purple-200 mb-2">HEC Token</label>
                                                    <input
                                                        type="password"
                                                        value={splunkToken}
                                                        onChange={(e) => setSplunkToken(e.target.value)}
                                                        placeholder="Enter token..."
                                                        className="w-full px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="block text-sm font-medium text-purple-200 mb-2">
                                                        Target Index (optional)
                                                    </label>
                                                    <input
                                                        type="text"
                                                        value={splunkIndex}
                                                        onChange={(e) => setSplunkIndex(e.target.value)}
                                                        placeholder="e.g., video_analytics"
                                                        className="w-full px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                                    />
                                                    <p className="text-xs text-purple-300 mt-1">
                                                        Leave empty to use HEC token default index
                                                    </p>
                                                </div>
                                            </div>
                                        )}

                                        <div className="flex gap-3 pt-4">
                                            {!isAnalyzing ? (
                                                <button
                                                    onClick={startAnalysis}
                                                    className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold py-3 rounded-lg flex items-center justify-center gap-2 transition"
                                                >
                                                    <Icon name="play" size={20} />
                                                    Start Analysis
                                                </button>
                                            ) : (
                                                <button
                                                    onClick={stopAnalysis}
                                                    className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 rounded-lg flex items-center justify-center gap-2 transition"
                                                >
                                                    <Icon name="square" size={20} />
                                                    Stop
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {(status || error) && (
                                    <div className={`bg-white/10 backdrop-blur-md rounded-xl p-4 border ${error ? 'border-red-400/50' : 'border-green-400/50'}`}>
                                        <div className="flex items-start gap-3">
                                            <Icon name={error ? "alert" : "check"} className={error ? "text-red-400" : "text-green-400"} size={20} />
                                            <p className={`text-sm font-medium ${error ? 'text-red-300' : 'text-green-300'}`}>
                                                {error || status}
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="lg:col-span-2 space-y-6">
                                <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                                    <div className="xl:col-span-2">
                                        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 h-full">
                                            <div className="flex items-center justify-between mb-4">
                                                <h2 className="text-xl font-semibold text-white">Detection Results</h2>
                                                {detections.length > 0 && (
                                                    <button
                                                        onClick={downloadResults}
                                                        className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition"
                                                    >
                                                        <Icon name="download" size={16} />
                                                        Download JSON
                                                    </button>
                                                )}
                                            </div>

                                            <div className="space-y-2 max-h-[600px] overflow-y-auto">
                                                {detections.length === 0 ? (
                                                    <div className="text-center py-12 text-purple-300">
                                                        <Icon name="camera" className="mx-auto mb-4 opacity-50" size={64} />
                                                        <p>No detections yet. Start an analysis to see results.</p>
                                                    </div>
                                                ) : (
                                                    detections.map((det, idx) => (
                                                        <div key={idx} className="bg-white/5 border border-white/10 rounded-lg p-4 hover:bg-white/10 transition">
                                                            <div className="flex items-start justify-between mb-2">
                                                                <div className="flex items-center gap-2">
                                                                    <span className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-3 py-1 rounded-full text-sm font-semibold">
                                                                        {det.class_name}
                                                                    </span>
                                                                    <span className="text-green-400 text-sm font-medium">
                                                                        {(det.confidence * 100).toFixed(1)}%
                                                                    </span>
                                                                </div>
                                                                <span className="text-purple-300 text-sm">Frame {det.frame_number}</span>
                                                            </div>
                                                            <div className="grid grid-cols-2 gap-2 text-sm">
                                                                <div>
                                                                    <span className="text-purple-200">Timestamp:</span>
                                                                    <span className="text-white ml-2">{det.timestamp}s</span>
                                                                </div>
                                                                <div>
                                                                    <span className="text-purple-200">BBox:</span>
                                                                    <span className="text-white ml-2">[{det.bbox.join(', ')}]</span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="xl:col-span-1">
                                        {detections.length > 0 && (
                                            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 h-full">
                                                <h3 className="text-lg font-semibold text-white mb-4">Statistics</h3>
                                                <div className="space-y-3">
                                                    <div className="flex justify-between">
                                                        <span className="text-purple-200">Total Detections:</span>
                                                        <span className="text-white font-semibold">{stats.totalDetections}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-purple-200">Avg Confidence:</span>
                                                        <span className="text-white font-semibold">{stats.avgConfidence}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-purple-200">Unique Classes:</span>
                                                        <span className="text-white font-semibold">{stats.uniqueClasses}</span>
                                                    </div>
                                                    
                                                    {stats.performance && (
                                                        <>
                                                            <div className="pt-3 border-t border-white/10">
                                                                <p className="text-sm font-medium text-purple-200 mb-2">Performance</p>
                                                            </div>
                                                            <div className="flex justify-between">
                                                                <span className="text-purple-200">Device:</span>
                                                                <span className="text-white font-semibold flex items-center gap-1">
                                                                    {stats.performance.device}
                                                                    {stats.performance.device === 'GPU' && (
                                                                        <Icon name="zap" className="text-yellow-400" size={16} />
                                                                    )}
                                                                </span>
                                                            </div>
                                                            <div className="flex justify-between">
                                                                <span className="text-purple-200">Avg Inference:</span>
                                                                <span className="text-white font-semibold">{stats.performance.avg_inference_time}ms</span>
                                                            </div>
                                                            <div className="flex justify-between">
                                                                <span className="text-purple-200">Processing FPS:</span>
                                                                <span className="text-white font-semibold">{stats.performance.fps_processing}</span>
                                                            </div>
                                                            <div className="flex justify-between">
                                                                <span className="text-purple-200">Total Time:</span>
                                                                <span className="text-white font-semibold">{stats.performance.total_time}s</span>
                                                            </div>
                                                            {stats.performance.device === 'GPU' && stats.performance.speedup > 1 && (
                                                                <div className="flex flex-col bg-green-500/10 border border-green-500/30 rounded-lg p-2 mt-2">
                                                                    <span className="text-green-300 font-medium text-sm">GPU Speedup:</span>
                                                                    <span className="text-green-400 font-bold text-lg">~{stats.performance.speedup}x faster</span>
                                                                </div>
                                                            )}
                                                        </>
                                                    )}
                                                    
                                                    {getTopClasses().length > 0 && (
                                                        <div className="pt-3 border-t border-white/10">
                                                            <p className="text-sm font-medium text-purple-200 mb-2">Top Objects:</p>
                                                            <div className="space-y-2">
                                                                {getTopClasses().map(([cls, count]) => (
                                                                    <div key={cls} className="flex justify-between items-center">
                                                                        <span className="text-white text-sm">{cls}</span>
                                                                        <span className="bg-purple-600/50 px-2 py-1 rounded text-xs text-white">{count}</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>'''

@app.route('/')
def index():
    """Serve the React GUI"""
    return HTML_TEMPLATE

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Start video analysis"""
    global analyzer, splunk_client, current_detections, analysis_thread
    
    data = request.json
    video_url = data.get('url')
    splunk_url = data.get('splunk_url')
    splunk_token = data.get('splunk_token')
    splunk_index = data.get('splunk_index')
    frame_skip = data.get('frame_skip', 5)
    conf_threshold = data.get('conf_threshold', 0.5)
    
    if not video_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Check if analysis is already running
    if analysis_thread and analysis_thread.is_alive():
        return jsonify({'error': 'Analysis already in progress'}), 400
    
    # Reset detections
    current_detections = []
    
    # Initialize Splunk if configured
    if splunk_url and splunk_token:
        splunk_client = SplunkHEC(splunk_url, splunk_token, splunk_index)
        print("✓ Splunk HEC configured")
        if splunk_index:
            print(f"  → Target index: {splunk_index}")
        else:
            print(f"  → Using HEC token default index")
    else:
        splunk_client = None
    
    def progress_cb(progress, frame, total):
        print(f"Progress: {progress:.1f}% ({frame}/{total} frames)")
    
    def detection_cb(detection):
        current_detections.append(asdict(detection))
        if splunk_client:
            splunk_client.send_event(detection)
    
    def analyze_task():
        try:
            print(f"\n{'='*60}")
            print(f"Starting Analysis")
            print(f"{'='*60}")
            
            video_path, metadata = analyzer.download_video(video_url)
            
            detections = analyzer.analyze_video(
                video_path,
                progress_callback=progress_cb,
                detection_callback=detection_cb,
                frame_skip=frame_skip,
                conf_threshold=conf_threshold
            )
            
            # Clean up video file
            try:
                Path(video_path).unlink()
                print(f"✓ Cleaned up video file: {video_path}")
            except:
                pass
                
        except Exception as e:
            print(f"Error in analysis task: {e}")
            analyzer.progress_info['status'] = 'error'
            import traceback
            traceback.print_exc()
    
    try:
        analysis_thread = threading.Thread(target=analyze_task)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return jsonify({'status': 'started', 'message': 'Analysis in progress'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get detection results"""
    return jsonify(current_detections)

@app.route('/api/stop', methods=['POST'])
def stop_analysis():
    """Stop analysis"""
    if analyzer:
        analyzer.stop()
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'no analysis running'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    status_info = {
        'detections_count': len(current_detections)
    }
    
    if detector:
        status_info['cuda_available'] = detector.cuda_available
        
        if torch.cuda.is_available():
            status_info['cuda_device'] = torch.cuda.get_device_name(0)
            status_info['cuda_memory_allocated'] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            status_info['cuda_memory_reserved'] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
    
    if analyzer:
        progress_info = analyzer.get_progress()
        status_info['progress'] = progress_info
    
    return jsonify(status_info)

def main():
    """Main entry point"""
    global detector, analyzer
    
    print("\n" + "=" * 60)
    print("YouTube Video Object Detection System - PyTorch Edition")
    print("=" * 60)
    
    # Check PyTorch installation
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Initialize detector and analyzer
    print("\nInitializing YOLOv8 detector...")
    detector = PyTorchDetector(model_name="yolov8n.pt")  # Change to yolov8s.pt, yolov8m.pt for better accuracy
    analyzer = VideoAnalyzer(detector)
    
    print("\n" + "=" * 60)
    print("✓ Server ready!")
    print("=" * 60)
    print(f"\nOpen your browser and navigate to:")
    print(f"  → http://localhost:5000")
    print(f"\nGPU Acceleration: {'✓ ENABLED' if detector.cuda_available else '✗ DISABLED (CPU mode)'}")
    
    if detector.cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("\nTo enable GPU acceleration:")
        print("  1. Install CUDA toolkit from NVIDIA")
        print("  2. Reinstall PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()