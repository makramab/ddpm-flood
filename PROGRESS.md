# DDPM Flood — Project Progress Tracker

**Project**: DDPM for Urban Flood Tipping-Point Discovery
**Target**: 7-day execution → presentation with actual results for Prof. Yuki Miura (CERA Lab, NYU Tandon)

---

## Day 1: Environment Setup + Data Acquisition

### Environment
- [x] Set up Python environment with uv (Python 3.11)
- [x] Install core DDPM dependencies (torch, einops, tqdm, pillow, numpy)
- [x] Install data preprocessing deps (geopandas, fiona, netCDF4, xarray, shapely, pyproj)
- [x] Install visualization deps (matplotlib, seaborn) — cartopy skipped for now

### Repos
- [x] Clone DDPM_REMD (Wang et al.): https://github.com/tiwarylab/DDPM_REMD
- [x] Clone GISSR (Miura et al.): https://github.com/ym2540/GIS_FloodSimulation
- [ ] Verify DDPM_REMD runs on AIB9 example (train ~1000 steps, confirm no errors)

### Data Acquisition
- [x] Download NACCS geodatabase — explored, has ARI data but NOT per-storm matrix
- [x] Download Sandy Inundation Zone from NYC Open Data
- [x] Download USGS High Water Marks for Sandy validation (347 records)
- [x] Download Battery tide gauge data for Sandy (peak 4.29m confirmed)
- [x] Download Sandy instrument data (69 sensor records)
- [ ] DesignSafe/CERA — registration pending approval
- [x] Download DeepSurge sample (1 file) — explored, too sparse for NYC (3-7 nodes)
- [x] Download Danso et al. ADCIRC Irma — verified: 9,728 NYC nodes, 4,438 with surge data
- [ ] **Download all 20 Danso et al. storms** (~11.3 GB) ← NEXT STEP

### Data Investigation (completed)
- [x] Explored NACCS geodatabase structure (16 layers, 16,326 save points, 1,050 storm tracks)
- [x] Discovered NACCS only has summary stats, not per-storm matrix
- [x] Investigated NACCS REST API — server is dead
- [x] Investigated CHS (Coastal Hazards System) — inaccessible
- [x] Evaluated DeepSurge on Zenodo — good storm count but sparse NYC coverage + novelty concern
- [x] Found Danso et al. ADCIRC dataset — 20 storms, real ADCIRC, excellent NYC coverage
- [x] Verified Irma test file: 1.8M mesh nodes, 169 nodes near Battery, surge data confirmed

---

## Day 2: Data Exploration + Preprocessing

### Download & Verify
- [ ] Download remaining 19 storms from Zenodo
- [ ] Verify all 20 storms have same mesh structure
- [ ] Check surge range at Battery across all 180 scenarios

### Preprocess
- [ ] Extract NYC metro nodes (bbox: 40.45-40.95 lat, -74.30 to -73.70 lon)
- [ ] Identify wet nodes (with valid surge data) across all scenarios
- [ ] Compute θ (peak surge at Battery) for each of 180 scenarios
- [ ] Create spatial patches (~220 patches of ~20 nodes each)
- [ ] Build training data matrix: ~39,600 rows × ~21 columns
- [ ] Normalize with StandardScaler, save scaler
- [ ] Save `flood_training_data.npy` + `scaler.pkl` + `patch_assignments.pkl`

---

## Days 3-4: DDPM Code Adaptation + Training

### Code Adaptation
- [ ] Adapt `run_training.py`: set `op_num=20`, `unmask_number=1`, data path, `train_num_steps` (50K-100K), `batch_size=64-128`
- [ ] Adapt `gen_sample.py`: set `op_num`, model checkpoint path, sample file path
- [ ] Confirm no changes needed in `denoising_diffusion_pytorch.py`

### Training
- [ ] Hold out 3-4 storms for validation
- [ ] Train DDPM on flood data (spatial patches)
- [ ] Monitor loss convergence
- [ ] If loss plateaus early, try reducing `dim` from 32 to 16

---

## Days 5-6: Experiments + Figure Generation

### Experiment 1 — Interpolation (hold-out storms)
- [ ] Generate patches for held-out storms
- [ ] Stitch patches into full NYC map
- [ ] Compare DDPM vs. ADCIRC ground truth (R², RMSE)
- [ ] Produce scatter plot (Figure 1)

### Experiment 2 — Sandy Extrapolation
- [ ] Set θ = 4.29m (Sandy Battery peak)
- [ ] Generate predicted NYC flood map
- [ ] Overlay with Sandy HWMs + inundation zone
- [ ] Produce comparison map (Figure 2)

### Experiment 3 — Tipping-Point Discovery (KEY RESULT)
- [ ] Generate at θ = 0.5, 0.6, ... 5.0m (sweep)
- [ ] Plot surge at key NYC locations vs. θ
- [ ] Identify sharp jumps → tipping points
- [ ] Produce response curves (Figure 3)

### Experiment 4 — Flood Map Panels
- [ ] Generate NYC flood maps at θ = 1.0, 2.0, 3.0, 4.0m
- [ ] Show progressive flooding (Figure 4)

### Validation & Plausibility
- [ ] Verify generated surge ≥ 0
- [ ] Check spatial consistency (nearby patches agree)
- [ ] Cross-scenario: predict SLR variants, compare to ADCIRC

### Comparison Table
- [ ] Produce DDPM vs. ADCIRC vs. GISSR comparison (Figure 5)

---

## Day 7: Presentation Assembly

- [ ] Build 12-15 slides
- [ ] Write speaker notes
- [ ] Prepare for potential questions from Prof. Miura

---

## Current Status

**Currently working on**: Day 1 — Downloading all 20 Danso et al. ADCIRC storms
**Blockers**: None (DesignSafe pending but no longer critical)
**Key decisions made**:
- Training data: Danso et al. ADCIRC (20 storms × 9 variants = 180 scenarios)
- Spatial patch approach to address row-to-column ratio (~39,600 rows × ~21 columns)
- Conditioning variable θ = peak surge at Battery
- Sandy validation via extrapolation (θ = 4.29m)
- See `EXPERIMENT_PLAN.md` for full details

**Notes**:
- Using uv for Python environment management (not conda)
- PyTorch 2.10.0 installed, MPS backend available (Apple Silicon), no CUDA
- Cloned repos are in `.gitignore` (reference only, not part of our repo)
- NACCS ARI data kept as supplementary spatial context
- DeepSurge explored but not used for training (novelty concern)
