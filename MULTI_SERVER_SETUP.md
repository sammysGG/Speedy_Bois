# Multi-Server RL Training Setup Guide

## 🎯 Overview

This guide will help you deploy the DCM RL Racing training system across multiple servers to run parallel hyperparameter experiments. Each server will train independently with different configurations, and the master dashboard will monitor all servers from one interface.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         Master Dashboard (This Server)         │
│     Shows stats from all servers via tabs      │
│            https://dcmrace.goathost.gg          │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼────┐   ┌─────▼──────┐  ┌────▼───────┐
│  Server 1  │   │  Server 2  │  │  Server 3  │
│   (Local)  │   │ 7950X3D #1 │  │ 7950X3D #2 │
│  9950X     │   │            │  │            │
│ gamma=0.99 │   │gamma=0.995 │  │gamma=0.999 │
└────────────┘   └────────────┘  └────────────┘
```

## 📋 Prerequisites

**Each remote server needs:**
- Ubuntu/Debian Linux (or similar)
- Python 3.8+
- 16+ CPU cores recommended (works on any CPU)
- 8GB+ RAM
- Network access to master server (or expose port 5000)
- Git installed

## 🚀 Quick Deployment (For GitHub Copilot)

### **On Each Remote Server:**

Run these commands to deploy the training system:

```bash
# 1. Clone the repository
cd ~
git clone https://github.com/sammysGG/Speedy_Bois.git
cd Speedy_Bois/dcm-reinforcement-learning/web-viewer

# 2. Install dependencies
pip3 install --upgrade pip
pip3 install gymnasium[box2d] stable-baselines3 flask opencv-python numpy

# 3. Create models directory
mkdir -p models/checkpoints

# 4. Test the server (should start Flask on port 5000)
python3 viewer.py
```

**Expected output:**
```
Running on CPUs: [0, 1, 2, 3, ... 31]
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
```

### **Configure Firewall (if needed):**

```bash
# Allow port 5000 from master server IP
sudo ufw allow from MASTER_SERVER_IP to any port 5000

# Or allow from anywhere (less secure)
sudo ufw allow 5000/tcp
```

### **Optional: Setup Systemd Service**

To auto-start on boot:

```bash
# Create service file
sudo nano /etc/systemd/system/racing-viewer.service
```

Paste this content:

```ini
[Unit]
Description=DCM RL Racing Viewer
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/Speedy_Bois/dcm-reinforcement-learning/web-viewer
ExecStart=/usr/bin/python3 /home/YOUR_USERNAME/Speedy_Bois/dcm-reinforcement-learning/web-viewer/viewer.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable racing-viewer
sudo systemctl start racing-viewer
sudo systemctl status racing-viewer
```

## 🎛️ Master Server Configuration

### **Edit remote_servers.json on Master Server:**

```bash
cd /root/dcm-reinforcement-learning/web-viewer
nano remote_servers.json
```

**Update with your server IPs:**

```json
{
  "servers": [
    {
      "id": "local",
      "name": "Server 1 (Local)",
      "url": "http://localhost:5000",
      "enabled": true,
      "specs": "AMD Ryzen 9 9950X (32 threads)"
    },
    {
      "id": "remote1",
      "name": "Server 2 (Remote)",
      "url": "http://192.168.1.100:5000",
      "enabled": true,
      "specs": "AMD Ryzen 9 7950X3D (32 threads)"
    },
    {
      "id": "remote2",
      "name": "Server 3 (Remote)",
      "url": "http://192.168.1.101:5000",
      "enabled": true,
      "specs": "AMD Ryzen 9 7950X3D (32 threads)"
    }
  ]
}
```

**Replace IPs with:**
- Public IPs if servers are on different networks
- Private IPs if on same network/VPN
- Domain names if configured

### **Restart Master Dashboard:**

```bash
sudo systemctl restart racing-viewer
```

## 🎮 Using the Multi-Server Dashboard

### **Access Dashboard:**
https://dcmrace.goathost.gg/viewer/

### **You will see:**
- Server tabs at the top: `[Server 1] [Server 2] [Server 3]`
- Green dot = Online, Red dot = Offline
- Click any tab to view that server's stats and video feed
- Start/stop training on each server independently

### **Running Experiments:**

**Recommended setup for competition:**

| Server | Algorithm | Gamma | Learning Rate | N_Envs | Timesteps | Purpose |
|--------|-----------|-------|---------------|--------|-----------|---------|
| Server 1 | PPO | 0.99 | 3e-4 | 24 | 500k | Short-term rewards |
| Server 2 | PPO | 0.995 | 3e-4 | 24 | 500k | Balanced (default) |
| Server 3 | PPO | 0.999 | 3e-4 | 24 | 500k | Long-term planning |

**Each will complete in ~12-15 minutes**

### **To Start Training on Each Server:**

1. Click the server tab
2. Set configuration:
   - Algorithm: PPO
   - Timesteps: 500000
   - N_Envs: 24 (or 8 if you want to watch video)
   - Learning Rate: 0.0003
   - Gamma: (varies per server)
   - **Uncheck "Show Multi-Panel Grid"** for max speed
3. Click "Start Training"
4. Switch to another server tab and repeat

## 📊 Monitoring

### **Dashboard Features:**
- **Real-time stats**: Episode count, rewards, FPS, timesteps
- **Progress bar**: Shows completion percentage
- **Checkpoint counter**: Auto-saves every 50k timesteps
- **Video feed**: Live view of car training (if render enabled)

### **Per-Server Monitoring:**
```bash
# SSH to remote server
ssh user@192.168.1.100

# Check CPU usage
top -u YOUR_USERNAME

# Check training logs
tail -f ~/Speedy_Bois/dcm-reinforcement-learning/web-viewer/training.log

# Check saved models
ls -lh ~/Speedy_Bois/dcm-reinforcement-learning/web-viewer/models/

# Check checkpoints
ls -lh ~/Speedy_Bois/dcm-reinforcement-learning/web-viewer/models/checkpoints/PPO_0/
```

## 📦 Collecting Models

### **After training completes:**

```bash
# On remote server
cd ~/Speedy_Bois/dcm-reinforcement-learning/web-viewer/models
ls -lh  # Find your trained model

# Download to master server
scp user@remote:/path/to/models/PPO_trained_500000.zip ./
```

### **Or use rsync:**
```bash
rsync -avz user@192.168.1.100:~/Speedy_Bois/dcm-reinforcement-learning/web-viewer/models/ ./remote_models/
```

## 🧪 Testing Models

To test a model's performance:

```bash
cd ~/Speedy_Bois/dcm-reinforcement-learning
python3 -c "
from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load('web-viewer/models/PPO_trained_500000')
env = gym.make('CarRacing-v3', continuous=True, render_mode='human')

for episode in range(5):
    obs, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(f'Episode {episode+1}: Reward = {total_reward:.2f}')

env.close()
"
```

## 🏆 Competition Submission

### **Compare all models:**
1. Test each model for 10 episodes
2. Record average reward
3. Submit the best performing model

### **Submission format:**
```
PPO_gamma_0995_500k.zip  ← Your best model
algorithm: PPO
continuous: True
```

## 🔧 Troubleshooting

### **Server shows offline:**
- Check firewall: `sudo ufw status`
- Test connection: `curl http://server-ip:5000/stats`
- Check service: `sudo systemctl status racing-viewer`

### **Training not starting:**
- Check logs: `journalctl -u racing-viewer -f`
- Verify dependencies: `pip3 list | grep -E "gymnasium|stable-baselines3"`
- Check CPU: `lscpu` (need at least 2 cores)

### **Low performance:**
- Increase n_envs: Try 16-24 for 7950X3D
- Disable rendering: Uncheck "Show Multi-Panel Grid"
- Check CPU affinity in viewer.py (line 550)

### **Port conflicts:**
- Change port in viewer.py: `app.run(host='0.0.0.0', port=5001)`
- Update remote_servers.json with new port

## 💡 Tips

1. **Start simple**: Test with 1 remote server first
2. **Use SSH keys**: Setup passwordless SSH for easier management
3. **Monitor resources**: Keep an eye on CPU/RAM usage
4. **Backup models**: Copy checkpoints periodically during long runs
5. **Network latency**: Dashboard updates every 2s from remote servers
6. **Different configs**: Vary gamma (0.99-0.999) first, then learning rate

## 🎯 Expected Performance

**Per 7950X3D server (32 threads):**
- 8 envs with video: ~13,000 steps/min
- 24 envs no video: ~40,000 steps/min

**Training times for 500k steps:**
- 8 envs: ~38 minutes
- 16 envs: ~20 minutes  
- 24 envs: ~12 minutes

## 📚 Additional Resources

- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [CarRacing-v3 Environment](https://www.gymlibrary.dev/environments/box2d/car_racing/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

---

**Ready to deploy? Start with one remote server and work your way up!** 🚀
