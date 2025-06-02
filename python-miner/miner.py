import hashlib
import time
import threading
import logging
import psutil
import json
from datetime import datetime

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA not available. Running on CPU only.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner.log'),
        logging.StreamHandler()
    ]
)

class BitcoinMiner:
    def __init__(self):
        self.running = False
        self.hash_rate = 0
        self.total_hashes = 0
        self.start_time = None
        self.threads = []
        self.gpu_enabled = CUDA_AVAILABLE
        self.lock = threading.Lock()
        
        # CPU cores for mining
        self.cpu_cores = psutil.cpu_count(logical=True)
        logging.info(f"Detected {self.cpu_cores} CPU cores")
        
        # Initialize GPU if available
        if self.gpu_enabled:
            try:
                self.init_gpu()
                logging.info("GPU initialization successful")
            except Exception as e:
                logging.error(f"GPU initialization failed: {str(e)}")
                self.gpu_enabled = False

    def init_gpu(self):
        if not CUDA_AVAILABLE:
            return

        # CUDA kernel for SHA-256 mining
        cuda_code = """
        __global__ void sha256_kernel(unsigned int *nonce, unsigned int *result) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            // Simple SHA-256 implementation for demonstration
            // In real mining, this would be more complex
            atomicAdd(result, 1);
        }
        """
        
        try:
            self.mod = SourceModule(cuda_code)
            self.sha256_kernel = self.mod.get_function("sha256_kernel")
            logging.info("CUDA kernel compiled successfully")
        except Exception as e:
            logging.error(f"CUDA kernel compilation failed: {str(e)}")
            self.gpu_enabled = False

    def cpu_mine(self, thread_id):
        """CPU mining function"""
        while self.running:
            try:
                # Simulate mining work (in real mining, this would be actual SHA-256 calculations)
                data = f"Bitcoin mining thread {thread_id} - {time.time()}"
                hashlib.sha256(data.encode()).hexdigest()
                
                with self.lock:
                    self.total_hashes += 1
                
                # Sleep to prevent CPU overload in this demo
                time.sleep(0.001)
            except Exception as e:
                logging.error(f"Error in CPU mining thread {thread_id}: {str(e)}")

    def gpu_mine(self):
        """GPU mining function"""
        if not self.gpu_enabled:
            return

        while self.running:
            try:
                # Allocate memory on GPU
                nonce = cuda.mem_alloc(4)
                result = cuda.mem_alloc(4)
                
                # Launch kernel
                self.sha256_kernel(
                    nonce,
                    result,
                    block=(256, 1, 1),
                    grid=(100, 1)
                )
                
                with self.lock:
                    self.total_hashes += 25600  # 256 * 100 threads
                
                # Sleep to prevent GPU overload in this demo
                time.sleep(0.001)
            except Exception as e:
                logging.error(f"Error in GPU mining: {str(e)}")
                break

    def start_mining(self):
        """Start the mining process"""
        if self.running:
            logging.warning("Mining is already running")
            return

        self.running = True
        self.start_time = time.time()
        self.total_hashes = 0
        logging.info("Starting mining operation")

        # Start CPU mining threads
        for i in range(self.cpu_cores):
            thread = threading.Thread(target=self.cpu_mine, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            logging.info(f"Started CPU mining thread {i}")

        # Start GPU mining if available
        if self.gpu_enabled:
            gpu_thread = threading.Thread(target=self.gpu_mine)
            gpu_thread.daemon = True
            gpu_thread.start()
            self.threads.append(gpu_thread)
            logging.info("Started GPU mining thread")

        # Start hash rate monitoring
        monitor_thread = threading.Thread(target=self.monitor_hash_rate)
        monitor_thread.daemon = True
        monitor_thread.start()
        self.threads.append(monitor_thread)

    def stop_mining(self):
        """Stop the mining process"""
        if not self.running:
            logging.warning("Mining is not running")
            return

        self.running = False
        logging.info("Stopping mining operation")
        
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        self.threads.clear()
        self.calculate_hash_rate()
        logging.info(f"Mining stopped. Final hash rate: {self.hash_rate:.2f} H/s")

    def monitor_hash_rate(self):
        """Monitor and update hash rate"""
        while self.running:
            self.calculate_hash_rate()
            time.sleep(1)

    def calculate_hash_rate(self):
        """Calculate current hash rate"""
        if self.start_time is None:
            return 0
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.hash_rate = self.total_hashes / elapsed_time

    def get_status(self):
        """Get current mining status"""
        return {
            "running": self.running,
            "hash_rate": f"{self.hash_rate:.2f}",
            "total_hashes": self.total_hashes,
            "gpu_enabled": self.gpu_enabled,
            "cpu_cores": self.cpu_cores,
            "uptime": time.time() - (self.start_time or time.time())
        }

if __name__ == "__main__":
    try:
        miner = BitcoinMiner()
        miner.start_mining()
        
        # Keep the main thread running
        while True:
            status = miner.get_status()
            logging.info(f"Mining Status: {json.dumps(status, indent=2)}")
            time.sleep(5)
            
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
        miner.stop_mining()
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        if miner.running:
            miner.stop_mining()
