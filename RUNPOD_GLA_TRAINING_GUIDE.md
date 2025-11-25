shoutout cursor

# RunPod Training Guide: GLA Model on RTX A6000

This guide walks you through training your GLA (Gated Linear Attention) model on RunPod using an RTX A6000 GPU.

## Prerequisites

- RunPod account with credits
- Local machine with Cursor/VSCode installed
- SSH key pair configured (see Step 1 below)

---

## Generate and Configure SSH Key

If you don't already have an SSH key, generate one locally:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "your_email@domain.com"
```

Copy your public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Add ssh public key to your [RunPod account settings](https://www.runpod.io/console/user/settings) under **SSH Public Keys**.

⚠️ **Important**: Add your SSH key BEFORE deploying your Pod. If you add it after, you'll need to manually inject it into the Pod.

---

## Deploy a RunPod Pod

1. Go to [RunPod Pods](https://www.runpod.io/console/pods)
2. Click **Deploy** or **+ GPU Pod**
3. Select your GPU:
   - Filter for **RTX A6000** (48GB VRAM)
   - Choose based on price/availability
4. Select a template:
   - **Recommended**: PyTorch 2.x template (official RunPod templates support SSH over TCP)
   - Make sure **SSH Terminal Access** checkbox is checked under **Instance Pricing**
5. Configure storage:
   - Container Disk: 50GB minimum
   - Volume Disk: 100GB+ recommended (persistent storage for checkpoints)
6. Click **Deploy**

Wait for the Pod to fully initialize (usually 1-2 minutes).

---

## Configure SSH Connection in Cursor

### Get Connection Details

1. From the [Pods page](https://www.runpod.io/console/pods), click on your deployed Pod
2. Click **Connect**
3. Look for **Direct TCP Ports** section:
   ```
   TCP port -> 69.48.159.6:25634 -> :22
   ```
   Note the IP address (`69.48.159.6`) and port (`25634`)

### Add to Cursor SSH Config

1. In Cursor, open Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Select **Remote-SSH: Connect to Host** → **Add New SSH Host**
3. This opens your SSH config file. Add:

```
Host runpod-gla-training
    HostName 69.48.159.6
    User root
    Port 25634
    IdentityFile ~/.ssh/id_ed25519
```

Replace the `HostName` and `Port` with your actual values from Step 3.

4. Save and close the file

---

## Connect to Your Pod

1. Open Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Select **Remote-SSH: Connect to Host**
3. Choose `runpod-gla-training` (or whatever name you used)
4. Cursor will open a new window and connect
5. When prompted, select **Linux** as the platform
6. Click **Open Folder** and navigate to `/workspace`

You're now connected! The `/workspace` directory is persistent storage.

---

## Set Up Your Project on RunPod

### Clone Your Repository

```bash
cd /workspace
git clone <your-repo-url> compute-aware-arch-search
cd compute-aware-arch-search
```

Or if using SSH:

```bash
git clone git@github.com:your-username/compute-aware-arch-search.git
```

### Install Dependencies

Your project uses `uv` for dependency management. Install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # Add uv to PATH
```

Install project dependencies:

```bash
cd /workspace/compute-aware-arch-search
uv sync
```

This will create a virtual environment and install all packages from `pyproject.toml`.

### Verify GPU Access

```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
```

Expected output:

```
GPU Available: True
GPU Name: NVIDIA RTX A6000
```

---

## Configure Weights & Biases (Optional but Recommended)

Login to W&B to track your training:

```bash
wandb login
```

Enter your W&B API key when prompted.

---

## Run Training

Your `train.py` is already configured to use GLA. The training will:

- Train 28 layers sequentially (one per Qwen3 layer)
- Save checkpoints to `linear_attention_checkpoints/`
- Log to W&B with separate runs for each layer
- Save final weights as safetensors

Start training:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run training
python train.py
```

### Monitor Progress

- **Terminal**: Watch the training output in real-time
- **W&B Dashboard**: View metrics, losses, and learning rate curves
- **Checkpoints**: Check `linear_attention_checkpoints/safetensors/` for saved weights

### Expected Training Time

With RTX A6000 and current settings:

- ~50-100 steps per layer (depends on max_steps calculation)
- Each layer takes ~10-30 minutes
- Total: ~5-14 hours for all 28 layers

---

## Monitor and Manage Training

### Check GPU Utilization

```bash
watch -n 1 nvidia-smi
```

### Resume if Training Stops

The script trains layers sequentially. Check `layer_results.csv` to see which layers completed:

```bash
cat layer_results.csv
```

If training stopped at layer N, you can modify `train.py` to skip completed layers by changing:

```python
for layer_idx in range(1, num_layers + 1):
```

to:

```python
for layer_idx in range(N+1, num_layers + 1):  # Resume from layer N+1
```

### Download Checkpoints

Once training completes (or periodically), download your checkpoints:

```bash
# From your local machine
scp -r -P 25634 root@69.48.159.6:/workspace/compute-aware-arch-search/linear_attention_checkpoints ./
```

Or use Cursor's file explorer to drag and drop files.

---

## Important Notes

### Checkpoint Directory

The script saves results to `linear_attention_checkpoints/` as specified in `train.py`:

```python
layer_output_dir = os.path.join("linear_attention_checkpoints", f"layer_{layer_idx}")
```

Final weights are saved to:

```
linear_attention_checkpoints/safetensors/student_layer_{N}.safetensors
```

### Pod Lifecycle

- **Stop vs Terminate**: Stopping preserves `/workspace`, terminating deletes everything
- **Port Changes**: If you stop/resume, the port number may change. Update your SSH config!
- **Billing**: You're charged per minute while running. Stop when not in use.

### Cost Optimization

- RTX A6000 typically costs $0.79-$1.19/hour on RunPod
- For 14 hours: ~$11-$17 total
- Stop the Pod immediately after training completes

---

## Troubleshooting

### Can't Connect via SSH

1. Verify Pod is running and fully initialized
2. Check that SSH key is in RunPod settings
3. Verify port numbers haven't changed
4. Try connecting via RunPod web terminal first

### CUDA Out of Memory

Reduce batch size in `train.py`:

```python
per_device_batch_size = 16  # Down from 32
```

### Training Loss Not Decreasing

- Check W&B for `normalized_loss` metric (should be < 1.0)
- Verify teacher model and student model are on correct devices
- Check that GLA config dimensions match Qwen3 hidden size (2048)

### Missing Dependencies

```bash
uv sync  # Reinstall all dependencies
```

---

## After Training

1. **Download all checkpoints** to your local machine
2. **Download `layer_results.csv`** for analysis
3. **Stop the Pod** to avoid charges
4. **Push code changes** to your git repository if needed

---

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Activate environment
source .venv/bin/activate

# Run training
python train.py

# Monitor training
tail -f nohup.out  # If running in background

# Check completed layers
cat layer_results.csv

# Download checkpoints (from local machine)
scp -r -P <PORT> root@<IP>:/workspace/compute-aware-arch-search/linear_attention_checkpoints ./
```

---

## Configuration Summary

**Current Setup:**

- Model: GLA (Gated Linear Attention)
- Teacher: Qwen3-1.7B
- Config: `linear_attn/gla_config.json`
- Batch Size: 32
- Learning Rate: 1e-3 → 1e-5 (cosine decay)
- Warmup: 10% of steps
- GPU: RTX A6000 (48GB VRAM)
- Checkpoints: `linear_attention_checkpoints/`

Ready to train! 🚀
