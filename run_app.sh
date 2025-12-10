#!/bin/bash

# Stock Dashboard Flask Application Runner
# This script sets up a virtual environment, installs dependencies, and runs the Flask app on port 5001

set -e  # Exit on any error

echo "🚀 Starting Stock Dashboard Setup..."

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

echo "🎯 Starting Flask application on port 5001..."
echo "🌐 Open your browser to: http://localhost:5001"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run the Flask app with port 5001 using the entry point script
python run_app.py --port 5001