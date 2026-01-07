# Password Protection Guide

## 🔒 Security Features

The dashboard now requires a password for all control actions while keeping viewing public.

### **Public (No Password Required)**
- ✅ View live stats and metrics
- ✅ Watch video feed / grid view
- ✅ See training progress
- ✅ Monitor all servers

### **Protected (Password Required)**
- 🔐 Start Training
- 🔐 Stop Training
- 🔐 Parallel Training

## 🔑 Default Password

```
speedybois2024
```

## 🛠️ Changing the Password

### **Method 1: Environment Variable (Recommended)**

```bash
# On your server
export RACING_PASSWORD="your_secure_password_here"

# Restart the service
sudo systemctl restart racing-viewer
```

### **Method 2: Systemd Service File**

```bash
sudo nano /etc/systemd/system/racing-viewer.service
```

Add this line under `[Service]`:
```ini
Environment="RACING_PASSWORD=your_secure_password_here"
```

Then reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart racing-viewer
```

### **Method 3: Edit Code Directly**

Edit `/root/dcm-reinforcement-learning/web-viewer/viewer.py`:

```python
# Line 23
CONTROL_PASSWORD = os.environ.get('RACING_PASSWORD', 'your_password_here')
```

## 🎮 Using the Dashboard

### **1. Viewing (No Password)**
- Simply open: https://dcmrace.goathost.gg/viewer/
- Click server tabs to view different servers
- Watch live training progress
- No authentication needed

### **2. Starting Training (Password Required)**
1. Click "Start Training" button
2. Password prompt appears: `🔒 Enter password to control training:`
3. Enter password: `speedybois2024`
4. Configure training settings
5. Confirm to start

### **3. Stopping Training (Password Required)**
1. Click "Stop Training" button
2. Password prompt appears
3. Enter password
4. Training stops gracefully

## ❌ Error Handling

**Wrong Password:**
```
❌ Invalid password
```
- HTTP 401 Unauthorized returned
- Training does not start
- Try again with correct password

**Network Issues:**
- Dashboard remains viewable
- Control actions may timeout
- Check server connectivity

## 🌐 Multi-Server Setup

Each server has its own password (configured independently):

- **Server 1 (Local)**: Uses this server's password
- **Server 2 (Remote)**: Uses Server 2's password
- **Server 3 (Remote)**: Uses Server 3's password

When switching server tabs, you'll need each server's password to control it.

## 🔐 Security Best Practices

1. **Change the default password** immediately
2. **Use strong passwords** (12+ characters, mixed case, numbers)
3. **Don't share passwords** via insecure channels
4. **Use environment variables** instead of hardcoding
5. **Restrict firewall** to only allow trusted IPs if possible

## 📝 API Endpoints

### **Public Endpoints (No Auth)**
```
GET  /                     - Dashboard UI
GET  /stats                - Current stats JSON
GET  /video_feed           - Live video stream
GET  /api/server_info      - Server information
GET  /api/remote_servers   - Remote server list
```

### **Protected Endpoints (Requires Password)**
```
POST /train               - Start training
POST /stop                - Stop training  
POST /train_parallel      - Start parallel training
```

**Request Format:**
```json
{
  "password": "speedybois2024",
  "algorithm": "PPO",
  "timesteps": 500000,
  ...
}
```

## 🚀 Quick Test

```bash
# Test without password (should fail)
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"PPO","timesteps":10000}'

# Response: {"status": "error", "message": "Invalid password"}

# Test with password (should work)
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"password":"speedybois2024","algorithm":"PPO","timesteps":10000}'

# Response: {"status": "training_started", ...}
```

## 💡 Tips

- **Password is in memory only** - not logged or stored
- **Browser doesn't cache password** - you'll need to re-enter each time
- **Password works per-action** - each button press requires it
- **Viewers don't need password** - perfect for public monitoring
- **Change default ASAP** - especially if exposed to internet

---

**Default Password:** `speedybois2024`

**Change it in:** `/etc/systemd/system/racing-viewer.service` or via `RACING_PASSWORD` env var
