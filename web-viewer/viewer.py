#!/usr/bin/env python3
"""
Live Car Racing Viewer with Training Integration
Streams the racing environment to a web interface with parallel training support
"""
import gymnasium
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import io
import base64
import os
import json

app = Flask(__name__)

class LiveViewerCallback(BaseCallback):
    """Callback to update viewer during training"""
    def __init__(self, viewer, training_id=0, verbose=0):
        super().__init__(verbose)
        self.viewer = viewer
        self.training_id = training_id
        self.last_frame_time = time.time()
        self.episode_rewards = []
        self.episode_steps = 0
        self.episode_reward = 0
        self.episodes_completed = 0
        
    def _on_step(self):
        # Check if we should stop training
        if self.viewer.should_stop:
            return False  # Stop training
        
        # Update current timestep count
        if self.training_id == 0:  # Only main training updates global stats
            self.viewer.stats['current_timesteps'] = self.num_timesteps
        
        # Update parallel training stats
        if self.training_id in self.viewer.parallel_stats:
            self.viewer.parallel_stats[self.training_id]['current_timesteps'] = self.num_timesteps
            self.viewer.parallel_stats[self.training_id]['episodes'] = self.episodes_completed
        
        # Get training environment rewards
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            self.episode_reward += reward
            self.episode_steps += 1
            
            # Update viewer stats (only for main training)
            if self.training_id == 0:
                self.viewer.stats['reward'] = float(reward)
                self.viewer.stats['step'] = self.episode_steps
                self.viewer.stats['total_reward'] = float(self.episode_reward)
                
                if reward > self.viewer.stats['max_reward']:
                    self.viewer.stats['max_reward'] = float(reward)
        
        # Check if episode ended
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self.episodes_completed += 1
            if self.training_id == 0:
                self.viewer.stats['episode'] = self.episodes_completed
            
            self.episode_rewards.append(self.episode_reward)
            
            # Update parallel stats
            if self.training_id in self.viewer.parallel_stats:
                self.viewer.parallel_stats[self.training_id]['total_reward'] = float(self.episode_reward)
                self.viewer.parallel_stats[self.training_id]['avg_reward'] = float(np.mean(self.episode_rewards[-10:]))
            
            self.episode_reward = 0
            self.episode_steps = 0
        
        # Update frame from training environment (about every 30ms for 30fps) - only for main training
        if self.training_id == 0 and time.time() - self.last_frame_time > 0.03:
            try:
                # Get the training environment
                env = self.training_env
                
                if hasattr(self.viewer, 'render_all') and self.viewer.render_all:
                    # Render all environments for multi-panel view
                    if hasattr(env, 'envs'):
                        num_envs = min(len(env.envs), 8)  # Max 8 panels
                        
                        with self.viewer.frame_lock:  # Prevent tearing
                            for idx in range(num_envs):
                                if idx < len(env.envs) and hasattr(env.envs[idx], 'render'):
                                    frame = env.envs[idx].render()
                                    if frame is not None:
                                        # Smaller frames for 8-panel grid
                                        size = (300, 225) if num_envs <= 4 else (200, 150)
                                        frame = cv2.resize(frame, size)
                                        # Add environment number label
                                        cv2.putText(frame, f"Car {idx+1}", (10, 20),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                        self.viewer.env_frames[idx] = frame
                            
                            # Combine frames into grid
                            valid_frames = [f for f in self.viewer.env_frames[:num_envs] if f is not None]
                            
                            if len(valid_frames) >= num_envs:
                                if num_envs <= 4:
                                    # 2x2 grid for 4 envs
                                    top_row = np.hstack([self.viewer.env_frames[0], self.viewer.env_frames[1]])
                                    bottom_row = np.hstack([self.viewer.env_frames[2], self.viewer.env_frames[3]])
                                    combined = np.vstack([top_row, bottom_row])
                                else:
                                    # 4x2 grid for 8 envs
                                    row1 = np.hstack([self.viewer.env_frames[0], self.viewer.env_frames[1], 
                                                     self.viewer.env_frames[2], self.viewer.env_frames[3]])
                                    row2 = np.hstack([self.viewer.env_frames[4], self.viewer.env_frames[5],
                                                     self.viewer.env_frames[6], self.viewer.env_frames[7]])
                                    combined = np.vstack([row1, row2])
                                
                                self.viewer.frame = combined.copy()  # Copy to prevent tearing
                                self.last_frame_time = time.time()
                else:
                    # Single environment render (original behavior)
                    if hasattr(env, 'envs'):
                        env = env.envs[0]
                    
                    if hasattr(env, 'render'):
                        frame = env.render()
                        if frame is not None:
                            frame = cv2.resize(frame, (800, 600))
                            with self.viewer.frame_lock:
                                self.viewer.frame = frame.copy()
                            self.last_frame_time = time.time()
            except Exception as e:
                pass  # Skip frame on error
        return True

class RacingViewer:
    def __init__(self):
        self.frame = None
        self.env = None
        self.model = None
        self.is_running = False
        self.is_training = False
        self.should_stop = False  # Flag to stop training
        self.start_time = None
        self.training_thread = None
        self.parallel_trainings = []  # List of parallel training threads
        self.parallel_stats = {}  # Stats for each parallel training
        self.stats = {
            'reward': 0,
            'total_reward': 0,
            'max_reward': 0,
            'episode': 0,
            'step': 0,
            'episode_time': 0,
            'training_mode': False,
            'total_timesteps': 0,
            'current_timesteps': 0,
            'algorithm': 'None',
            'parallel_count': 0,
            'checkpoints_saved': 0
        }
        
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            # Try to detect algorithm from path
            if 'DQN' in model_path or 'dqn' in model_path:
                self.model = DQN.load(model_path)
                self.stats['algorithm'] = 'DQN'
            elif 'PPO' in model_path or 'ppo' in model_path:
                self.model = PPO.load(model_path)
                self.stats['algorithm'] = 'PPO'
            elif 'SAC' in model_path or 'sac' in model_path:
                self.model = SAC.load(model_path)
                self.stats['algorithm'] = 'SAC'
            elif 'A2C' in model_path or 'a2c' in model_path:
                self.model = A2C.load(model_path)
                self.stats['algorithm'] = 'A2C'
            else:
                # Try each one
                try:
                    self.model = DQN.load(model_path)
                    self.stats['algorithm'] = 'DQN'
                except:
                    try:
                        self.model = PPO.load(model_path)
                        self.stats['algorithm'] = 'PPO'
                    except:
                        self.model = SAC.load(model_path)
                        self.stats['algorithm'] = 'SAC'
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def start_environment(self, continuous=False, seed=69):
        """Start the racing environment"""
        self.env = gymnasium.make("CarRacing-v3", render_mode='rgb_array', continuous=continuous)
        self.env.reset(seed=seed)
        
    def train_model(self, algorithm, continuous, timesteps, learning_rate=None, gamma=None, 
                   exploration_fraction=None, n_envs=4, training_id=0, model_name=None, render_all=False):
        """Train a new model with vectorized environments and checkpointing"""
        if training_id == 0:  # Main training
            self.is_training = True
            self.should_stop = False
            self.stats['training_mode'] = True
            self.stats['algorithm'] = algorithm
            self.stats['total_timesteps'] = timesteps
            self.stats['current_timesteps'] = 0
            self.stats['episode'] = 0
            self.stats['step'] = 0
            self.stats['reward'] = 0
            self.stats['total_reward'] = 0
            self.stats['max_reward'] = 0
            self.stats['checkpoints_saved'] = 0
            self.start_time = time.time()
            self.render_all = render_all
            self.env_frames = [None] * n_envs  # Store frames from all environments
            self.frame_lock = threading.Lock()  # Prevent screen tearing
        
        # Initialize parallel stats
        if training_id not in self.parallel_stats:
            self.parallel_stats[training_id] = {
                'algorithm': algorithm,
                'total_timesteps': timesteps,
                'current_timesteps': 0,
                'episodes': 0,
                'total_reward': 0,
                'avg_reward': 0,
                'status': 'training',
                'gamma': gamma,
                'learning_rate': learning_rate
            }
        
        # Create vectorized environment (4 parallel environments)
        print(f"🚀 Creating {n_envs} parallel environments for training {training_id}...")
        
        # Render mode: if render_all, render all envs, otherwise only first
        render_mode = 'rgb_array' if (training_id == 0 and render_all) or training_id == 0 else None
        
        env = make_vec_env(
            "CarRacing-v3",
            n_envs=n_envs,
            seed=69,
            env_kwargs={'continuous': continuous, 'render_mode': render_mode}
        )
        
        # Set default hyperparameters if not provided
        if algorithm == "DQN":
            lr = learning_rate if learning_rate else 1e-4
            g = gamma if gamma else 0.99
            ef = exploration_fraction if exploration_fraction else 0.3
            self.model = DQN(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                learning_rate=lr,
                gamma=g,
                exploration_fraction=ef,
                seed=69,
            )
            print(f"🎮 DQN Config [{training_id}]: lr={lr}, gamma={g}, exploration={ef}, n_envs={n_envs}")
        elif algorithm == "PPO":
            lr = learning_rate if learning_rate else 3e-4
            g = gamma if gamma else 0.99
            self.model = PPO(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                learning_rate=lr,
                gamma=g,
                n_steps=2048,  # Optimized for racing
                batch_size=64,
                seed=69,
            )
            print(f"🎮 PPO Config [{training_id}]: lr={lr}, gamma={g}, n_envs={n_envs}, n_steps=2048")
        elif algorithm == "A2C":
            lr = learning_rate if learning_rate else 7e-4
            g = gamma if gamma else 0.99
            self.model = A2C(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                learning_rate=lr,
                gamma=g,
                seed=69,
            )
            print(f"🎮 A2C Config [{training_id}]: lr={lr}, gamma={g}, n_envs={n_envs}")
        
        # Setup checkpointing (save every 50k timesteps)
        checkpoint_dir = f"./models/checkpoints/{model_name or algorithm}_{training_id}/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=50000 // n_envs,  # Adjust for vectorized envs
            save_path=checkpoint_dir,
            name_prefix=f"{algorithm}_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        
        # Custom callback for live updates
        viewer_callback = LiveViewerCallback(self, training_id=training_id)
        
        # Train with checkpoints and live updates
        try:
            print(f"🏁 Starting training [{training_id}]: {timesteps} timesteps with {n_envs} parallel envs...")
            self.model.learn(
                total_timesteps=timesteps,
                callback=[checkpoint_callback, viewer_callback],
                progress_bar=False
            )
            
            # Only save final model if training completed successfully (not stopped early)
            if not self.should_stop:
                os.makedirs("./models", exist_ok=True)
                final_name = model_name if model_name else f"{algorithm}_trained_{timesteps}"
                model_path = f"./models/{final_name}"
                self.model.save(model_path)
                print(f"✅ Model [{training_id}] saved to {model_path}")
                
                # Count checkpoints
                checkpoints = len([f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')])
                if training_id == 0:
                    self.stats['checkpoints_saved'] = checkpoints
            else:
                print(f"⚠️ Training [{training_id}] stopped by user")
        except Exception as e:
            print(f"❌ Training error [{training_id}]: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if training_id == 0:
                self.is_training = False
                self.should_stop = False
                self.stats['training_mode'] = False
                self.stats['current_timesteps'] = 0
                self.stats['episode_time'] = int(time.time() - self.start_time) if self.start_time else 0
                self.is_running = False
            
            # Update parallel stats
            if training_id in self.parallel_stats:
                self.parallel_stats[training_id]['status'] = 'completed' if not self.should_stop else 'stopped'
            
            env.close()
    
    def run_episode(self, deterministic=True):
        """Run one episode and stream frames"""
        if self.env is None:
            self.start_environment()
            
        obs, info = self.env.reset()
        self.start_time = time.time()
        self.stats['total_reward'] = 0
        self.stats['step'] = 0
        self.stats['episode'] += 1
        self.stats['max_reward'] = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and self.is_running:
            # Get action from model or random
            if self.model:
                action, _ = self.model.predict(obs, deterministic=deterministic)
            else:
                action = self.env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Update stats
            self.stats['reward'] = reward
            self.stats['total_reward'] += reward
            self.stats['step'] += 1
            self.stats['episode_time'] = int(time.time() - self.start_time)
            
            # Track max reward
            if reward > self.stats['max_reward']:
                self.stats['max_reward'] = reward
            
            # Get frame
            frame = self.env.render()
            if frame is not None:
                # Resize for web
                frame = cv2.resize(frame, (800, 600))
                self.frame = frame
            
            time.sleep(0.03)  # ~30 FPS
            
    def get_frame(self):
        """Get current frame as JPEG"""
        with self.frame_lock if hasattr(self, 'frame_lock') else threading.Lock():
            if self.frame is None:
                # Create blank frame
                blank = np.zeros((600, 800, 3), dtype=np.uint8)
                text = 'Training...' if self.is_training else 'Waiting for episode...'
                cv2.putText(blank, text, (250, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

viewer = RacingViewer()

@app.route('/')
def index():
    """Main viewer page - new dashboard"""
    return render_template('dashboard.html')

@app.route('/old')
def old_index():
    """Old viewer page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = viewer.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Get current stats"""
    return jsonify(viewer.stats)

@app.route('/start/<int:continuous>')
def start(continuous):
    """Start a new episode"""
    if not viewer.is_running and not viewer.is_training:
        viewer.is_running = True
        viewer.start_environment(continuous=bool(continuous))
        threading.Thread(target=viewer.run_episode, daemon=True).start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/stop')
def stop():
    """Stop the current episode or training"""
    viewer.is_running = False
    viewer.should_stop = True
    
    # Give it a moment to stop gracefully
    if viewer.is_training:
        return jsonify({'status': 'stopping', 'message': 'Training will stop at next checkpoint'})
    return jsonify({'status': 'stopped'})

@app.route('/train', methods=['POST'])
def train():
    """Start training a model with custom hyperparameters"""
    if viewer.is_training or viewer.is_running:
        return jsonify({'status': 'error', 'message': 'Already running'})
    
    data = request.json
    algorithm = data.get('algorithm', 'PPO')
    continuous = data.get('continuous', False)
    timesteps = data.get('timesteps', 50000)
    learning_rate = data.get('learning_rate')
    gamma = data.get('gamma')
    exploration_fraction = data.get('exploration_fraction')
    n_envs = data.get('n_envs', 4)  # Number of parallel environments
    render_all = data.get('render_all', False)  # Render all environments
    
    # Start training in background
    viewer.is_running = True
    viewer.training_thread = threading.Thread(
        target=viewer.train_model,
        args=(algorithm, continuous, timesteps, learning_rate, gamma, exploration_fraction, n_envs, 0, None, render_all),
        daemon=True
    )
    viewer.training_thread.start()
    
    return jsonify({
        'status': 'training_started', 
        'algorithm': algorithm, 
        'timesteps': timesteps,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'n_envs': n_envs,
        'render_all': render_all
    })

@app.route('/train_parallel', methods=['POST'])
def train_parallel():
    """Start parallel training with multiple configurations"""
    if viewer.is_training or viewer.is_running:
        return jsonify({'status': 'error', 'message': 'Already running'})
    
    data = request.json
    configs = data.get('configs', [])
    
    if not configs or len(configs) > 4:
        return jsonify({'status': 'error', 'message': 'Must provide 1-4 training configurations'})
    
    # Clear previous parallel stats
    viewer.parallel_stats = {}
    viewer.parallel_trainings = []
    viewer.is_training = True
    viewer.stats['parallel_count'] = len(configs)
    
    # Start each training in its own thread
    for idx, config in enumerate(configs):
        thread = threading.Thread(
            target=viewer.train_model,
            args=(
                config['algorithm'],
                config['continuous'],
                config['timesteps'],
                config.get('learning_rate'),
                config.get('gamma'),
                config.get('exploration_fraction'),
                config.get('n_envs', 2),  # Fewer envs per parallel training
                idx,
                config.get('name')
            ),
            daemon=True
        )
        thread.start()
        viewer.parallel_trainings.append(thread)
    
    return jsonify({
        'status': 'parallel_training_started',
        'count': len(configs),
        'configs': [f"{c['algorithm']} (gamma={c.get('gamma', 0.99)})" for c in configs]
    })

@app.route('/parallel_stats')
def parallel_stats():
    """Get stats for all parallel trainings"""
    return jsonify(viewer.parallel_stats)

@app.route('/load_model/<path:model_path>')
def load_model(model_path):
    """Load a trained model"""
    success = viewer.load_model(model_path)
    return jsonify({'status': 'loaded' if success else 'error'})

@app.route('/api/server_info')
def server_info():
    """Get server information for multi-server dashboard"""
    import platform
    import socket
    
    # Get CPU info
    try:
        cpu_info = os.popen('lscpu | grep "Model name"').read().strip()
        cpu_name = cpu_info.split(':')[1].strip() if ':' in cpu_info else 'Unknown CPU'
    except:
        cpu_name = 'Unknown CPU'
    
    # Get CPU count
    cpu_count = os.cpu_count() or 0
    
    # Get hostname
    hostname = socket.gethostname()
    
    return jsonify({
        'hostname': hostname,
        'cpu_name': cpu_name,
        'cpu_count': cpu_count,
        'platform': platform.system(),
        'stats': viewer.stats,
        'parallel_stats': viewer.parallel_stats,
        'is_training': viewer.is_training,
        'is_running': viewer.is_running
    })

@app.route('/api/remote_servers')
def get_remote_servers():
    """Get list of configured remote servers"""
    try:
        with open('remote_servers.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'servers': []})

if __name__ == '__main__':
    # Set CPU affinity to cores 8-31
    try:
        os.sched_setaffinity(0, range(8, 32))
        print(f"Running on CPUs: {sorted(os.sched_getaffinity(0))}")
    except:
        print("Could not set CPU affinity (may not be supported)")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
