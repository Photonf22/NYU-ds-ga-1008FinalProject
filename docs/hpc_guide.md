# HPC Cloud Bursting Guide (Fall 2025)
This document summarizes the HPC Cloud Bursting resources that are now available for CSCI-GA
2572. Please share it with your team and teaching assistants. Reach out to Shenglong or the NYU
HPC team if you encounter any onboarding issues; we can schedule a Zoom walkthrough as needed.

### Quick Overview
* Do not run course workloads on Greene GPUs; they are reserved for research use.
* Every student and TA has access to NYU’s HPC Cloud Bursting environment with preemptible GPU capacity.
* The Cloud Bursting Open OnDemand portal is at ood-burst-001.hpc.nyu.edu (NYU VPN re-
quired when off campus).
Accounts and Quotas
* Slurm account: csci ga 2572-2025fa
* Allocation: 300 GPU hours (18,000 minutes) per student, plus ample CPU time
* Allowed partitions: interactive, n2c48m24, g2-standard-12, g2-standard-24, g2-standard-48,
c12m85-a100-1, c24m170-a100-2, n1s8-t4-1

### Access Instructions
1. Log in to the Greene cluster: NYU HPC Access Instructions
2. From a Greene login node, connect to the burst environment:
ssh burst
3. On the log-burst node, request interactive resources with srun (examples below).
You can alternatively launch sessions directly from the Open OnDemand portal without a separate
SSH hop.

#### Launching Jobs
* CPU-only interactive shell for 4 hours:
```bash
srun --account=csci_ga_2572-2025fa --partition=interactive --time=04:00:00 --pty /bin/bash
```
* L4 GPU (partition g2-standard-12) for 4 hours:
```bash
srun --account=csci_ga_2572-2025fa --partition=g2-standard-12 --gres=gpu:1 --time=04:00:00 --pty /bin/bash
```
* A100 GPU (partition c12m85-a100-1) for 4 hours:
```bash
srun --account=csci_ga_2572-2025fa --partition=c12m85-a100-1 --gres=gpu --time=04:00:00 --pty /bin/bash
```
Feel free to adjust wall-time, partition, and GPU counts within the allocation and partition limits

#### Spot Instance Guidance
* Cloud Bursting runs on Google Cloud spot instances (preemptible VMs); instances may termi-
nate unexpectedly.
* Enable checkpointing in your training scripts and write state to /scratch/<NetID>.
* Add the following directive to batch scripts so jobs auto-requeue after preemption:

#### Batch
```bash
SBATCH --requeue
```
* Review Google’s documentation for spot instances: Spot VMs and Preemptible VMs.
Data Transfer
* Greene data transfer node hostname: greene-dtn
* Example: copy the public Singularity image to your Cloud Burst session
```bash
scp -rp greene-dtn:/scratch/work/public/singularity/ubuntu-20.04.3.sif .
```
* Use standard NYU HPC data transfer practices for large datasets.
Singularity and Conda
* Overlay templates: /share/apps/overlay-fs-ext3
* Singularity images: /share/apps/images
* Detailed walkthrough for Singularity + Miniconda overlays: Singularity with Miniconda
Support and Next Steps
* Ask TAs to validate the workflow (SSH + OOD) before sharing broadly with students.
* Direct logistics questions to Campuswire; escalate cluster issues to the HPC team.
* For lingering setup problems, contact Shenglong to schedule support
