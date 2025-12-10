#!/usr/bin/env python3
"""
Entry point for running the Flask application.
This ensures proper imports work when running from different directories.
"""

import sys
import os

# Add the parent directory to Python path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import and run the app
from src.app import app

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the Stock Dashboard Flask app')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    args = parser.parse_args()

    print(f"🎯 Starting Flask application on port {args.port}...")
    print(f"🌐 Open your browser to: http://localhost:{args.port}")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    app.run(debug=True, host='0.0.0.0', port=args.port)