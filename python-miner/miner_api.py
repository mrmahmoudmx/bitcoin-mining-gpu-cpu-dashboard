from flask import Flask, jsonify, request
from flask_cors import CORS
from miner import BitcoinMiner
import logging
from collections import deque
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

miner = BitcoinMiner()
# Store performance history (last 60 data points)
performance_history = deque(maxlen=60)
history_lock = threading.Lock()

def update_performance_history():
    """Update performance history every second"""
    while True:
        if miner.running:
            with history_lock:
                performance_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'hash_rate': float(miner.hash_rate),
                    'total_hashes': miner.total_hashes
                })
        threading.Event().wait(1)  # Update every second

# Start history tracking thread
history_thread = threading.Thread(target=update_performance_history, daemon=True)
history_thread.start()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner_api.log'),
        logging.StreamHandler()
    ]
)

@app.route('/status', methods=['GET'])
def get_status():
    """Get current mining status"""
    try:
        status = miner.get_status()
        return jsonify({
            "success": True,
            "data": status
        })
    except Exception as e:
        logging.error(f"Error getting status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/control', methods=['POST'])
def control_miner():
    """Control mining operations"""
    try:
        action = request.json.get('action')
        
        if action == 'start':
            if not miner.running:
                miner.start_mining()
                msg = "Mining started successfully"
            else:
                msg = "Mining is already running"
            
        elif action == 'stop':
            if miner.running:
                miner.stop_mining()
                msg = "Mining stopped successfully"
            else:
                msg = "Mining is not running"
                
        else:
            return jsonify({
                "success": False,
                "error": "Invalid action. Use 'start' or 'stop'"
            }), 400

        return jsonify({
            "success": True,
            "message": msg,
            "status": miner.get_status()
        })

    except Exception as e:
        logging.error(f"Error controlling miner: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get mining performance history"""
    try:
        with history_lock:
            return jsonify({
                "success": True,
                "data": list(performance_history)
            })
    except Exception as e:
        logging.error(f"Error getting history: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "gpu_available": miner.gpu_enabled,
        "cpu_cores": miner.cpu_cores
    })

if __name__ == '__main__':
    try:
        logging.info("Starting Mining API server...")
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Failed to start API server: {str(e)}")
