# RunPod Setup for CALDERA 72B Compression

## Requirements
- 2x A100 80GB (or equivalent 160GB+ total VRAM)
- 300GB storage for artifacts + model cache
- SSH access for remote execution

## API Setup

```bash
# API key location
~/.runpod/config.toml

# Key format
apikey = "rpa_XXXX..."
```

## Network Volume Creation

Network volumes persist across pod restarts. Required for long compression jobs.

```bash
# GraphQL mutation to create network volume
curl -s -X POST https://api.runpod.io/graphql \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_KEY" \
  -d '{"query": "mutation { createNetworkVolume(input: { name: \"caldera-4bit\", size: 300, dataCenterId: \"CA-MTL-3\" }) { id name size dataCenterId } }"}'
```

**Issue encountered**: Datacenter `US-CA-1` returned "Failed to find data center".
**Fix**: Use `CA-MTL-3` instead.

Current network volume: `6iyh6aj5gd` (300GB, CA-MTL-3)

## Pod Creation

### Critical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `gpuTypeId` | `NVIDIA A100 80GB PCIe` | Full ID required |
| `gpuCount` | `2` | For 72B model |
| `networkVolumeId` | `6iyh6aj5gd` | Our 300GB volume |
| `volumeMountPath` | `/workspace` | **REQUIRED** - without this, container fails |
| `startSsh` | `true` | Enable SSH |
| `ports` | `22/tcp` | Expose SSH port |
| `cloudType` | `COMMUNITY` | More availability than SECURE |

### Working Pod Creation Command

```bash
RUNPOD_KEY="rpa_XXXX" && curl -s -X POST https://api.runpod.io/graphql \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_KEY" \
  -d '{"query": "mutation { podFindAndDeployOnDemand( input: { cloudType: COMMUNITY, gpuCount: 2, gpuTypeId: \"NVIDIA A100 80GB PCIe\", networkVolumeId: \"6iyh6aj5gd\", volumeMountPath: \"/workspace\", containerDiskInGb: 30, name: \"caldera-4bit-72b\", imageName: \"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\", startSsh: true, ports: \"22/tcp\" } ) { id costPerHr machineId desiredStatus } }"}'
```

### Issues Encountered

#### 1. Volume Mount Error
**Error**: `invalid mount config for type "volume": field Target must not be empty`
**Cause**: Missing `volumeMountPath` parameter
**Fix**: Add `volumeMountPath: "/workspace"` to mutation

#### 2. SSH Not Working
**Error**: `Your SSH client doesn't support PTY` when using RunPod proxy SSH
**Cause**: `startSsh: true` alone doesn't expose the SSH port
**Fix**: Add `ports: "22/tcp"` to expose SSH directly

#### 3. Pod Stuck Provisioning
**Symptom**: `runtime: null` persists for minutes, pod never starts
**Cause**: GPU unavailability in selected datacenter/cloud type
**Potential fixes**:
- Try `COMMUNITY` instead of `SECURE` cloud
- Try different GPU types (H100, A100 SXM)
- Try different datacenters
- Wait for availability

#### 4. Network Volume Not Attached
**Symptom**: `/workspace` shows local volume, not network volume
**Cause**: Used `volumeInGb` (local) instead of `networkVolumeId`
**Fix**: Use `networkVolumeId: "6iyh6aj5gd"` and remove `volumeInGb`

#### 5. Network Volume Datacenter Lock
**Symptom**: "No instances available" even when GPUs exist elsewhere
**Cause**: Network volumes are datacenter-specific. Volume in CA-MTL-3 can only attach to pods in CA-MTL-3.
**Fix**: Either:
  - Wait for GPU availability in the volume's datacenter
  - Delete volume and recreate in datacenter with GPU availability
  - Use local volume instead (data won't persist across pod restarts)

**Workaround used (2025-01-07)**:
- CA-MTL-3 had no GPU availability
- Created network volume in US-CA-2 (`tyd7umfoe1`) - also no GPU availability
- Tested US-TX-3, US-KS-2 - network volumes work but no GPUs
- US-GA-1 - no storage clusters for network volumes
- **Conclusion**: Community cloud A100 80GB availability is in datacenters that don't support network volumes
- **Solution**: Use local 300GB volume. Sync artifacts back before terminating pod.

Network volumes created (for future use):
- `6iyh6aj5gd` - CA-MTL-3 (300GB)
- `tyd7umfoe1` - US-CA-2 (300GB)

## Checking Pod Status

```bash
# Get pod details including SSH port
curl -s -X POST https://api.runpod.io/graphql \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_KEY" \
  -d '{"query": "{ pod(input: { podId: \"POD_ID\" }) { id runtime { uptimeInSeconds ports { ip isIpPublic publicPort privatePort type } } } }"}'
```

## SSH Connection

Once pod is running with SSH exposed:
```bash
# Connection format (from API response)
ssh -p PUBLIC_PORT root@PUBLIC_IP -i ~/.ssh/id_ed25519

# Example
ssh -p 12514 root@104.255.9.187 -i ~/.ssh/id_ed25519
```

## Environment Setup on Pod

```bash
# 1. Install rsync (not included by default)
apt-get update && apt-get install -y rsync

# 2. Sync project code
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
  -e "ssh -p PORT -i ~/.ssh/id_ed25519" \
  /local/path/qwen2-caldera-max/ \
  root@IP:/workspace/qwen2-caldera-max/

# 3. Source environment setup (redirects all caches to /workspace)
cd /workspace/qwen2-caldera-max
source scripts/runpod_setup.sh

# 4. Install dependencies
pip install -e .

# 5. Start health monitor
./scripts/runpod_health_monitor.sh &
```

## Running Compression

```bash
# Generate calibration data
python scripts/generate_calibration.py --config configs/qwen2_72b_caldera_4bit_selective.yaml

# Run compression (late-layer-first, skips layers 70-79)
python scripts/compress.py --config configs/qwen2_72b_caldera_4bit_selective.yaml
```

## Cleanup

```bash
# Terminate pod
curl -s -X POST https://api.runpod.io/graphql \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_KEY" \
  -d '{"query": "mutation { podTerminate(input: { podId: \"POD_ID\" }) }"}'

# Network volume persists - don't delete unless done with project
```

## Cost Tracking

| Resource | Cost |
|----------|------|
| 2x A100 80GB PCIe (Community) | ~$2.38/hr |
| 300GB Network Volume | ~$0.07/hr |
| **Total** | ~$2.45/hr |

Expected compression time: 8-12 hours = ~$20-30
