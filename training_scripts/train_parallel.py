#!/usr/bin/env python3
"""
Parallel RL Training with CPU Affinity
Trains multiple algorithms simultaneously on cores 8-31
"""
import os
import gymnasium
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import multiprocessing as mp
import sys

# Set CPU affinity to cores 8-31 (24 threads)
os.sched_setaffinity(0, range(8, 32))

# Verify
print(f"Process running on CPUs: {os.sched_getaffinity(0)}")
print(f"Available CPUs for training: {len(os.sched_getaffinity(0))}")

def train_model(config):
    """Train a single model with given configuration"""
    algo_name = config['algorithm']
    continuous = config['continuous']
    total_timesteps = config['timesteps']
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"Starting {algo_name} ({'continuous' if continuous else 'discrete'})")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = gymnasium.make("CarRacing-v3", render_mode=None, continuous=continuous)
    env.reset(seed=69)  # Fixed seed for competition
    
    # Create eval environment
    eval_env = gymnasium.make("CarRacing-v3", render_mode=None, continuous=continuous)
    eval_env = Monitor(eval_env)
    
    # Model save paths
    model_path = f"./models/{run_id}_{algo_name}_{'cont' if continuous else 'disc'}"
    log_path = f"./logs/{run_id}_{algo_name}_{'cont' if continuous else 'disc'}"
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=model_path,
        name_prefix=f"{algo_name}_checkpoint"
    )
    
    # Create model based on algorithm
    try:
        if algo_name == "DQN":
            if continuous:
                print(f"⚠️  DQN doesn't support continuous - skipping")
                return
            model = DQN(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                learning_rate=config.get('learning_rate', 1e-4),
                gamma=config.get('gamma', 0.99),
                exploration_fraction=config.get('exploration_fraction', 0.3),
                seed=69,
            )
        
        elif algo_name == "PPO":
            model = PPO(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                learning_rate=config.get('learning_rate', 3e-4),
                gamma=config.get('gamma', 0.99),
                n_steps=config.get('n_steps', 2048),
                batch_size=config.get('batch_size', 64),
                seed=69,
            )
        
        elif algo_name == "SAC":
            if not continuous:
                print(f"⚠️  SAC requires continuous actions - skipping")
                return
            model = SAC(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                learning_rate=config.get('learning_rate', 3e-4),
                gamma=config.get('gamma', 0.99),
                seed=69,
            )
        
        elif algo_name == "A2C":
            model = A2C(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                learning_rate=config.get('learning_rate', 7e-4),
                gamma=config.get('gamma', 0.99),
                seed=69,
            )
        
        # Train
        print(f"\n🚀 Starting training for {algo_name}...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        final_path = f"{model_path}/final_model"
        model.save(final_path)
        print(f"\n✅ {algo_name} training complete! Saved to {final_path}")
        
    except Exception as e:
        print(f"\n❌ Error training {algo_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Training configurations
    configs = [
        {
            'algorithm': 'DQN',
            'continuous': False,
            'timesteps': 200_000,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'exploration_fraction': 0.3,
        },
        {
            'algorithm': 'PPO',
            'continuous': False,
            'timesteps': 200_000,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'n_steps': 2048,
            'batch_size': 64,
        },
        {
            'algorithm': 'PPO',
            'continuous': True,
            'timesteps': 200_000,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'n_steps': 2048,
            'batch_size': 64,
        },
        {
            'algorithm': 'A2C',
            'continuous': False,
            'timesteps': 200_000,
            'learning_rate': 7e-4,
            'gamma': 0.99,
        },
    ]
    
    print("\n" + "="*60)
    print("🏎️  DCM Car Racing - Parallel Training")
    print("="*60)
    print(f"CPU Cores Available: {len(os.sched_getaffinity(0))}")
    print(f"Experiments to run: {len(configs)}")
    print(f"Running on CPUs: {sorted(os.sched_getaffinity(0))}")
    print("="*60 + "\n")
    
    # Ask which experiment to run
    if len(sys.argv) > 1:
        exp_num = int(sys.argv[1])
        if 0 <= exp_num < len(configs):
            train_model(configs[exp_num])
        else:
            print(f"Invalid experiment number. Choose 0-{len(configs)-1}")
    else:
        print("Available experiments:")
        for i, cfg in enumerate(configs):
            print(f"  {i}: {cfg['algorithm']} ({'continuous' if cfg['continuous'] else 'discrete'}) - {cfg['timesteps']:,} steps")
        print(f"\nUsage: python3 {sys.argv[0]} <experiment_number>")
        print("Or run all in parallel:")
        print(f"  taskset -c 8-15 python3 {sys.argv[0]} 0 &")
        print(f"  taskset -c 16-23 python3 {sys.argv[0]} 1 &")
        print(f"  taskset -c 24-27 python3 {sys.argv[0]} 2 &")
        print(f"  taskset -c 28-31 python3 {sys.argv[0]} 3 &")

if __name__ == "__main__":
    main()
