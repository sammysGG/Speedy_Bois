#!/bin/bash
# Launch parallel training with CPU affinity
# Cores 8-31 (24 threads) split into 4 groups of 6 threads each

echo "🏎️  Starting Parallel RL Training"
echo "Using cores 8-31 (24 threads) in 4 groups"
echo ""

cd /root/dcm-reinforcement-learning/training_scripts

# DQN Discrete - Cores 8-13
echo "Starting DQN (discrete) on cores 8-13..."
taskset -c 8-13 python3 train_parallel.py 0 > ../logs/dqn_discrete.log 2>&1 &
DQN_PID=$!
echo "  PID: $DQN_PID"

sleep 5

# PPO Discrete - Cores 14-19
echo "Starting PPO (discrete) on cores 14-19..."
taskset -c 14-19 python3 train_parallel.py 1 > ../logs/ppo_discrete.log 2>&1 &
PPO_DISC_PID=$!
echo "  PID: $PPO_DISC_PID"

sleep 5

# PPO Continuous - Cores 20-25
echo "Starting PPO (continuous) on cores 20-25..."
taskset -c 20-25 python3 train_parallel.py 2 > ../logs/ppo_continuous.log 2>&1 &
PPO_CONT_PID=$!
echo "  PID: $PPO_CONT_PID"

sleep 5

# A2C Discrete - Cores 26-31
echo "Starting A2C (discrete) on cores 26-31..."
taskset -c 26-31 python3 train_parallel.py 3 > ../logs/a2c_discrete.log 2>&1 &
A2C_PID=$!
echo "  PID: $A2C_PID"

echo ""
echo "✅ All training jobs launched!"
echo ""
echo "Monitor progress:"
echo "  DQN:         tail -f ../logs/dqn_discrete.log"
echo "  PPO (disc):  tail -f ../logs/ppo_discrete.log"
echo "  PPO (cont):  tail -f ../logs/ppo_continuous.log"
echo "  A2C:         tail -f ../logs/a2c_discrete.log"
echo ""
echo "Check processes:"
echo "  ps aux | grep train_parallel"
echo ""
echo "Kill all training:"
echo "  pkill -f train_parallel"
