#!/bin/bash

# Text Generation Playground - Startup Script
# This script starts both the backend and frontend servers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Text Generation Playground ===${NC}"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Check if backend dependencies are installed
echo -e "${YELLOW}Checking backend dependencies...${NC}"
cd backend
if ! python3 -c "import fastapi, uvicorn, transformers" 2>/dev/null; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip3 install -r requirements.txt
else
    echo -e "${GREEN}✓ Backend dependencies installed${NC}"
fi
cd ..

# Check if ports are available
if check_port 8000; then
    echo -e "${RED}Error: Port 8000 is already in use${NC}"
    echo "Please stop the process using port 8000 or change the backend port"
    exit 1
fi

if check_port 8080; then
    echo -e "${YELLOW}Warning: Port 8080 is already in use${NC}"
    echo "Frontend will not be served, but you can open frontend/index.html directly"
    SERVE_FRONTEND=0
else
    SERVE_FRONTEND=1
fi

# Start backend server
echo
echo -e "${GREEN}Starting backend server on http://localhost:8000${NC}"
cd backend
python3 server.py &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Error: Backend failed to start${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
    echo -n "."
done
echo

# Start frontend server if port is available
if [ $SERVE_FRONTEND -eq 1 ]; then
    echo -e "${GREEN}Starting frontend server on http://localhost:8080${NC}"
    cd frontend
    python3 -m http.server 8080 &
    FRONTEND_PID=$!
    cd ..
    
    echo
    echo -e "${GREEN}=== Application Started Successfully ===${NC}"
    echo
    echo -e "Backend API:    ${GREEN}http://localhost:8000${NC}"
    echo -e "API Docs:       ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "Frontend:       ${GREEN}http://localhost:8080${NC}"
    echo
else
    echo
    echo -e "${GREEN}=== Backend Started Successfully ===${NC}"
    echo
    echo -e "Backend API:    ${GREEN}http://localhost:8000${NC}"
    echo -e "API Docs:       ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "Frontend:       Open ${GREEN}frontend/index.html${NC} in your browser"
    echo
fi

# Function to cleanup on exit
cleanup() {
    echo
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    if [ $SERVE_FRONTEND -eq 1 ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✓ Servers stopped${NC}"
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

echo -e "${YELLOW}Press Ctrl+C to stop the servers${NC}"
echo

# Wait for processes
wait
