#!/bin/bash
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
MODEL_DIR="./models/vosk-model-en-us-0.42-gigaspeech"
LLAMA_DIR="./build/bin/llama-server"
cd $PROJECT_DIR

trap "kill 0" EXIT

# Check for diagnostics flag
if [ "$1" == "--diagnostics" ]; then
    echo "=========================================="
    echo "Running Gesture Diagnostics..."
    echo "=========================================="
    echo "Press 'q' in the camera window to quit."
    python3 gestures/gesture_diagnostics.py
    exit 0
fi

echo "=========================================="
echo "Checking for dependencies..."
echo "=========================================="
echo ""

if [ ! -d "$MODEL_DIR" ]; then
   echo "Speech model not found, downloading now"
   echo ""
   mkdir -p models
   curl -L https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip -o vosk.zip && unzip vosk.zip -d models/
   echo "Done downloading speech model!"
fi

#python vosk/asr_server.py models/vosk-model-en-us-0.42-gigaspeech/ &


if [ ! -f "$LLAMA_DIR" ]; then
   echo "Llama.cpp not found, downloading now..."
   echo ""
   if [[ $(uname) == "Linux" ]]; then
      echo "Linux detected"
      curl -L https://github.com/ggml-org/llama.cpp/releases/download/b6945/llama-b6945-bin-ubuntu-vulkan-x64.zip -o llama.zip && unzip llama.zip
   else
      echo "MacOS detected"
      curl -L https://github.com/ggml-org/llama.cpp/releases/download/b6945/llama-b6945-bin-macos-x64.zip -o llama.zip && unzip llama.zip
   fi
   echo "Done downloading Llama.cpp!"
fi

echo "Installing python dependencies"
pip install -q -r requirements.txt

# Terminal 1: VLM Server (llama-server)
echo "✓ Opening Terminal 1: VLM Server (llama-server, version b6804)..."
echo '========================================'
echo 'VLM SERVER (Port 8080)'
echo '========================================'
echo ''
./build/bin/llama-server -hf Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 2>/dev/null 1>/dev/null &
sleep 5

echo "=========================================="
echo "Starting client components..."
echo "=========================================="
echo ""

# PDF Server
echo "✓ Starting PDF Server (try.pdf)..."
echo '========================================'
echo 'PDF SERVER (Port 9002)'
echo '========================================'
echo ''
python src/presenter/pdf_server.py $1 1>/dev/null &
sleep 2

# Orchestrator
echo "✓ Starting Orchestrator..."
echo '========================================'
echo 'ORCHESTRATOR (Port 9001)'
echo '========================================'
echo ''
python src/orchestrator/orchestrator.py $1 &
sleep 2

# Audio client
echo "✓ Starting Audio Client (Speech-to-Text)..."
echo '========================================'
echo 'AUDIO SERVER (Port 2700)'
echo '========================================'
echo ''
python src/audio/audio.py models/vosk-model-en-us-0.42-gigaspeech $1 1>/dev/null &
sleep 2

# Gesture Server
echo "✓ Starting Gesture Server..."
echo '========================================'
echo 'GESTURE SERVER (Port 9003)'
echo '========================================'
echo ''
python gestures/gesture_server.py 1>/dev/null &
sleep 2

# Open the main interface
echo "✓ Opening Unified Interface in default browser..."
open web/unified_interface.html

echo "=========================================="
echo "All systems are go!"
echo "=========================================="
wait
