# DDPM for Urban Flood Tipping-Point Discovery — Project Handoff Document

**Project**: PhD Application Assessment for Prof. Yuki Miura (CERA Lab, NYU Tandon)
**Date**: February 24, 2026
**Status**: Ready for implementation (7-day execution plan)

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Background: The Two Key Papers](#2-background-the-two-key-papers)
3. [The Microsoft Aurora Question](#3-the-microsoft-aurora-question)
4. [Thought Process & Key Questions](#4-thought-process--key-questions)
5. [Final Chosen Approach](#5-final-chosen-approach)
6. [Conceptual Mapping (Wang et al. → Floods)](#6-conceptual-mapping)
7. [Data Sources & Availability](#7-data-sources--availability)
8. [DDPM_REMD Code Structure & Adaptation Guide](#8-ddpm_remd-code-structure--adaptation-guide)
9. [Compute & Python Environment](#9-compute--python-environment)
10. [Data Preprocessing Pipeline](#10-data-preprocessing-pipeline)
11. [Execution Plan (7 Days)](#11-execution-plan-7-days)
12. [Expected Outputs & Figures](#12-expected-outputs--figures)
13. [Presentation Structure](#13-presentation-structure)
14. [References & External Links](#14-references--external-links)

---

## 1. Project Context

### The PI
- **Prof. Yuki Miura** — Assistant Professor, Dept. of Mechanical & Aerospace Engineering + Center for Urban Science and Progress (CUSP), NYU Tandon
- **Lab**: Climate, Energy, and Risk Analytics (CERA) Lab — [yukimiura.org](https://yukimiura.org/)
- **Contact**: yuki.miura@nyu.edu | +1 646-997-0534 | 370 Jay St, 13th Floor, Room 1309, Brooklyn, NY 11201
- **Affiliations**: Faculty Advisory Board at NYU Stern Volatility and Risk Institute; Member of NYC Panel on Climate Change (NPCC5)
- **Background**: PhD in Civil Engineering from Columbia University; previously at Morgan Stanley in climate risk management and quantitative strategy (2021-2024)

### The Assignment
Prof. Miura assigned this as a PhD application assessment project. The task: apply the DDPM (Denoising Diffusion Probabilistic Model) framework from Wang et al. 2022 to urban climate disaster research, specifically flood disasters. She provided two papers:
1. Wang et al. 2022 — "From Data to Noise to Data" (the DDPM methodology)
2. Miura et al. 2021 — GISSR (her own high-speed flood simulator)

She also mentioned that her lab previously attempted applying DDPM to hurricane-related events but stopped when Microsoft published the Aurora model, which covers similar territory at the atmospheric level.

### Timeline
- This project has been ongoing for several months (primarily conceptual research so far)
- Prof. Miura expects **results**, not just a proposal
- Target: **1 week** to produce a presentation with actual DDPM results

---

## 2. Background: The Two Key Papers

### 2.1 Wang et al. 2022 — "From Data to Noise to Data for Mixing Physics Across Temperatures with Generative Artificial Intelligence"

**Paper location**: `/Users/afrai/Documents/NYU/CERA_application/wang-et-al-2022-from-data-to-noise-to-data-for-mixing-physics-across-temperatures-with-generative-artificial (1).pdf`

**Published**: PNAS, Vol. 119, No. 32, August 2022
**Authors**: Yihang Wang, Lukas Herron, Pratyush Tiwary (University of Maryland)
**Code**: https://github.com/tiwarylab/DDPM_REMD

#### Core Idea
The paper's key insight is NOT just "use DDPM." It is a specific conceptual move:
- **Treat the control parameter (temperature T) as a fluctuating random variable**, not a fixed input
- This allows combining data from Replica Exchange Molecular Dynamics (REMD) simulations at temperatures T₁...Tₖ into a single joint distribution p(x, T)
- DDPM then learns this joint distribution and can generate samples at **any** temperature — including temperatures never simulated

#### What DDPM Does (Its 5 Unique Strengths)
1. **Discovers states never seen in training data** — transition states and metastable states absent from the training data at the temperature of interest (Fig. 3, black arrows)
2. **Extrapolates beyond the training range** — they removed the 400K replica entirely and DDPM still generated accurate configurations at 400K by extrapolating from 412K (Fig. 4)
3. **Learns from sparse, high-dimensional data without knowing what matters** — no need to pre-specify slow degrees of freedom; worked on 18 dihedral angles without being told which residues mattered
4. **Provides distributional outputs with built-in plausibility checks** — generated samples carry Boltzmann weights; unphysical "hallucinations" have negligibly low weights
5. **Pure postprocessing** — doesn't require modifying the simulator; works on existing data

#### Technical Details
- Architecture: U-Net with sinusoidal position embedding (Fig. 1 in paper)
- Input: s = {x, T} where x = molecular configuration (dihedral angles), T = instantaneous effective temperature
- 1000 discrete diffusion steps
- Adam optimizer, learning rate 2×10⁻⁵, EMA decay rate 0.995
- Demonstrated on: AIB9 peptide (18 dihedral angles, 10 REMD replicas, 100ns) and GACC RNA (24 dihedral angles, 48 replicas, 250ns)

### 2.2 Miura et al. 2021 — "High-Speed GIS-Based Simulation of Storm Surge-Induced Flooding Accounting for Sea Level Rise"

**Paper location**: `/Users/afrai/Documents/NYU/CERA_application/miura-et-al-2021-high-speed-gis-based-simulation-of-storm-surge-induced-flooding-accounting-for-sea-level-rise (1).pdf`

**Published**: Natural Hazards Review, Vol. 22, No. 3, 2021
**Authors**: Yuki Miura, Kyle T. Mandli, George Deodatis (Columbia University)
**Code**: https://github.com/ym2540/GIS_FloodSimulation

#### What GISSR Does
- **GIS-based Subdivision-Redistribution (GISSR)** methodology for coastal urban flood simulation
- Combines GIS with Manning's equation to calculate floodwater volume, then redistributes over the area
- Input: storm surge + tide time histories along coastline, topography (DEM), protective measures (seawalls)
- Output: time history of flood height at every point within the geographic area
- **Speed**: 0.01 seconds per simulation on a single CPU
- **Accuracy**: Validated against Hurricane Sandy actual inundation data in Lower Manhattan (16.7% total area difference)
- Can account for: sea level rise (SLR), seawalls (modeled as weirs), various protective measures

#### Key Results from the Paper
- Lower Manhattan divided into 18 divisions, each with a coastline segment
- Tested scenarios: Hurricane Sandy, Sandy + SLR 2050, Sandy + SLR 2100
- Tested seawall configurations: Big U Compartment C1, C1-C3, 1m full coastline wall, 2m full coastline wall
- Finding: Only a full-length seawall provides effective protection; partial walls (Big U C1 alone) provide zero reduction due to cascading around edges

---

## 3. The Microsoft Aurora Question

**Link**: https://news.microsoft.com/source/features/ai/microsofts-aurora-ai-foundation-model-goes-beyond-weather-forecasting/

Prof. Miura's lab stopped their earlier DDPM-for-hurricanes work because Microsoft Aurora covers similar territory. However, Aurora does NOT compete with our proposed approach:

| Feature | Aurora | Our DDPM Approach |
|---|---|---|
| Resolution | 0.25° (~28km) | Block-level (~10-500m) |
| Predicts | Atmospheric variables (weather, hurricanes, air quality) | Urban flood inundation at infrastructure level |
| Models seawalls/levees? | No | Yes (via training data from ADCIRC) |
| Urban-scale flooding? | No | Yes |
| Evaluates protective strategies? | No | Yes (tipping points for infrastructure failure) |

**Key argument**: Aurora predicts weather. It does NOT tell you "if surge reaches 3.2m and SLR is 0.4m, the seawall at Division 13 in Lower Manhattan fails and the East Village floods." That's what our DDPM does.

---

## 4. Thought Process & Key Questions

This section documents the questions that arose during our project scoping discussions and how we resolved them.

### Question 1: "Should GISSR be the training data source for DDPM?"

**Initial thought**: Train DDPM on GISSR-generated flood maps.

**Problem identified**: GISSR runs in 0.01 seconds. If you want a flood map at a new parameter, just run GISSR. There is no computational bottleneck for DDPM to solve. Training DDPM on GISSR output is training a neural network to approximate an already-fast, already-cheap calculator. The DDPM cannot discover physics that GISSR doesn't contain.

**Contrast with Wang et al.**: In their paper, REMD simulations were **expensive and incomplete** — that's what justified DDPM. The analog in floods must also be expensive simulation data, not GISSR.

**Resolution**: Use ADCIRC simulation data (expensive, high-fidelity) as training data instead. GISSR serves as a benchmark comparison, not the data source.

### Question 2: "Should the output be flood maps?"

**Initial thought**: Generate 2D flood inundation maps.

**Problems identified**:
- Scarcity of real flood map data (only a handful of well-documented events)
- Different cities have different topographies, making cross-city generalization difficult
- Full 2D maps are very high-dimensional, requiring enormous training data

**Resolution**: The output doesn't need to be a full flood map. It can be **flood depths at critical infrastructure points** (subway stations, power plants, hospitals) — a much lower-dimensional representation that is:
- More tractable for DDPM training
- More directly useful for risk assessment
- More transferable across contexts

However, for the proof-of-concept, we use whatever representation the NACCS data provides (peak surge at ~19,000 save points, extracted to the NYC subdomain).

### Question 3: "What is the actual novelty?"

**Rejected framing**: "We use GISSR as input data for DDPM" — this is just applying a rule-based model within a different framework, not novel.

**Accepted framing**: The novelty is in what DDPM uniquely enables:
1. **Tipping-point discovery** — discovering critical storm parameter thresholds where flood behavior changes dramatically (analogous to Wang et al.'s transition states). Neither ADCIRC (too expensive to sample densely) nor GISSR (too simplified) can do this efficiently.
2. **Extrapolation to unprecedented extremes** — generating plausible flood predictions for storms beyond the training range (the 500-year storm you never simulated)
3. **Distributional outputs** — uncertainty quantification for flood risk, not just deterministic point estimates
4. **ADCIRC fidelity at GISSR speed** — once trained, DDPM generates ADCIRC-quality predictions in seconds

### Question 4: "Why not just run more ADCIRC simulations?"

**Answer**: ADCIRC takes hours per simulation. To densely sample the storm parameter space (e.g., varying surge height × SLR × tide phase × storm track × forward speed), you'd need tens of thousands of runs. At hours each, that's computationally prohibitive. DDPM trained on ~1,050 NACCS storms can generate at any untested parameter combination instantly.

---

## 5. Final Chosen Approach

### One-Sentence Summary
> Train a DDPM on ~1,050 ADCIRC storm surge simulations from NACCS, conditioned on storm parameters, to generate ADCIRC-fidelity flood predictions at NYC save points for untested storm scenarios — enabling tipping-point discovery, extrapolation to unprecedented extremes, and probabilistic flood risk characterization.

### Architecture
```
Input to DDPM:  s = {surge_at_NYC_points, θ}
                where θ = storm parameters (intensity, forward speed,
                          track angle, landfall location, etc.)

DDPM learns:    p(surge_at_NYC_points, θ)
                the joint distribution over flood response and storm conditions

Output:         Samples from p(surge | θ) at any θ, including:
                - θ values not in training set (interpolation)
                - θ values beyond training range (extrapolation)
                - Dense sampling across θ-space (tipping-point discovery)
```

### Why This Is the Direct Analog to Wang et al.
| Wang et al. (molecular) | Our Approach (flood) |
|---|---|
| Temperature T (fluctuating random variable) | Storm parameter θ (surge height, SLR, intensity) |
| Molecular configuration x (dihedral angles) | Flood response (surge at NYC save points) |
| REMD replicas at T₁...Tₖ (expensive) | ADCIRC simulations at θ₁...θₖ (expensive) |
| Metastable/transition states | Flood tipping points (seawall overtopping thresholds) |
| DDPM generates at unsimulated T | DDPM generates at unsimulated θ |
| Boltzmann weights filter hallucinations | Physical constraints filter unphysical flood scenarios |

### Role of Each Tool
| Tool | Role |
|---|---|
| **ADCIRC (via NACCS)** | Training data source (expensive, high-fidelity, sparse) |
| **DDPM** | Learns joint distribution, generates at untested parameters, discovers tipping points |
| **GISSR** | Benchmark comparison (fast but simplified) |
| **Real observations (Sandy)** | Validation ground truth |

---

## 6. Conceptual Mapping

### DDPM's 5 Strengths → Flood Applications

| DDPM Strength | Flood Application |
|---|---|
| Discovers unseen states | Find critical flood tipping points — parameter combinations where the system transitions from "safe" to "catastrophic" that no simulation explicitly tested |
| Extrapolates beyond training range | Predict behavior under unprecedented storms — beyond historical record, beyond what was simulated |
| Learns from sparse data without priors | Works with limited expensive simulations (~1,050 ADCIRC runs), doesn't require knowing in advance which parameter combinations are dangerous |
| Distributional outputs + plausibility | Uncertainty quantification — not "the flood depth will be 1.5m" but "here's the distribution of possible outcomes" |
| Pure postprocessing | Can be applied to any existing flood simulation dataset without modifying the simulator |

---

## 7. Data Sources & Availability

### 7.1 Primary Training Data

#### NACCS ADCIRC Simulations (PRIORITY — Start Here)
- **What**: 1,050 synthetic tropical storms + 100 extratropical storms simulated with ADCIRC+SWAN
- **Coverage**: ~19,000 save points along the North Atlantic coast (Virginia to Maine), including NYC area
- **Data fields**: Peak storm surge and wind at each save point for each storm, plus storm track parameters
- **Format**: File geodatabase (GIS) + API
- **Download**:
  - Geodatabase: https://www.northeastoceancouncil.org/naccs/
  - ArcGIS MapServer: https://services.northeastoceandata.org/arcgis1/rest/services/NACCS/NACCS/MapServer
  - ScienceBase: https://www.sciencebase.gov/catalog/item/5a845a9be4b00f54eb34d501
  - NACCS main report (PDF): https://apps.dtic.mil/sti/tr/pdf/ADA621343.pdf
- **Access**: Free for research
- **Why this is the right data**: 1,050 storms with systematically varied parameters (track, intensity, forward speed, landfall) — designed to span the parameter space. This is the direct analog to Wang et al.'s REMD ladder of temperatures.

#### CERA/DesignSafe ADCIRC Hindcasts (Backup / Supplementary)
- **What**: 70 real historical storms (2003-2024) with full ADCIRC NetCDF outputs
- **Includes**: Sandy, Katrina, Harvey, Ian, Milton, and ~65 more
- **Format**: NetCDF (full spatiotemporal water level, wind, wave fields) + GeoTIFF (max inundation)
- **Download**: https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3932
- **Access**: Free for research (requires DesignSafe registration)
- **Historical storm archive viewer**: https://historicalstorms.coastalrisk.live/

### 7.2 Validation Data

#### Sandy Inundation Zone (NYC)
- **What**: Polygon shapefile of actually flooded areas during Sandy (2012)
- **Resolution**: 10m
- **Download**: https://data.cityofnewyork.us/Environment/Sandy-Inundation-Zone/uyj8-7rv5
- **Format**: Shapefile, GeoJSON

#### USGS High Water Marks
- **What**: ~1,200+ point observations of maximum water elevation during Sandy
- **Download**: https://stn.wim.usgs.gov/stndataportal/ (also via https://stn.wim.usgs.gov/FEV/)
- **Format**: CSV, shapefiles, JSON via API

#### NOAA Battery Tide Gauge (Station 8518750)
- **What**: 100+ years of water level data, including Sandy surge time series (peak ~3.4m at Battery)
- **Download**: https://tidesandcurrents.noaa.gov/waterlevels.html?id=8518750
- **API**: https://api.tidesandcurrents.noaa.gov/api/prod/
- **Format**: CSV, JSON, NetCDF

#### NYC FloodNet Sensors
- **What**: Real-time per-minute flood depth from 87 street-level sensors across NYC
- **Note**: This is an NYU CUSP/Tandon project — Prof. Miura likely has connections
- **Download**: https://www.floodnet.nyc/ and https://dataviz.floodnet.nyc/
- **Format**: CSV, API

### 7.3 Supporting / Topographic Data

| Dataset | URL | Format |
|---|---|---|
| NYC 1-ft DEM (LiDAR) | https://data.cityofnewyork.us/City-Government/1-foot-Digital-Elevation-Model-DEM-/dpc8-z3jc | GeoTIFF |
| NYC Building Footprints | https://data.cityofnewyork.us/Housing-Development/Building-Footprints/nqwf-w8eh | Shapefile/GeoJSON |
| NPCC SLR Projections | https://climateassessment.nyc/npcc-datasets/ | CSV/GIS |
| NYC Projected Sea Level Rise | https://data.cityofnewyork.us/City-Government/Projected-Sea-Level-Rise/6an6-9htp | GIS/tabular |
| NYC Stormwater Flood Maps | https://data.cityofnewyork.us/Environment/NYC-Stormwater-Flood-Maps/9i7c-xyvv | GIS layers |
| NOAA SLR Viewer (NY) | https://coast.noaa.gov/slrdata/Sea_Level_Rise_Vectors/NY/index.html | Shapefiles/GeoTIFF |
| FEMA National Flood Hazard Layer | https://www.fema.gov/flood-maps/national-flood-hazard-layer | Shapefiles |
| Hurricane Evacuation Zones (NYC) | https://data.cityofnewyork.us/Public-Safety/Hurricane-Evacuation-Zones/uihr-hn7s | Shapefile/GeoJSON |
| FEMA NFIP Insurance Claims (2M+) | https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2 | CSV/API |
| NYC flood data inventory (curated) | https://github.com/mebauer/nyc-flood-data | Index/links |

### 7.4 Supplementary Storm Catalogs (For Parameter Space Definition)

| Dataset | # Storms | What | URL |
|---|---|---|---|
| RAFT synthetic TC tracks | 40,000 (+ 900K CMIP6 extension) | Full TC tracks with intensity parameters | https://zenodo.org/records/10392725 |
| DeepSurge | 900,000 storms | Peak surge at 1,100 coastal points | https://doi.org/10.5281/zenodo.15021868 |
| STORM global synthetic TCs | 10,000 years equivalent | TC tracks for all global basins | https://data.4tu.nl/articles/dataset/STORM_IBTrACS_present_climate_synthetic_tropical_cyclone_tracks/12706085 |
| NOAA SLOSH MOM/MEOW | 100,000+ hypothetical storms (composited) | Max surge envelopes by category | https://www.nhc.noaa.gov/surge/momAvail.php |

### 7.5 Satellite-Derived Flood Datasets (Optional, For Future Work)

| Dataset | Resolution | Events | URL |
|---|---|---|---|
| Sen1Floods11 | 10m SAR | 11 events, 4,831 chips | https://github.com/cloudtostreet/Sen1Floods11 |
| UrbanSARFloods | 10m SAR | 18 events, 8,879 chips | https://github.com/jie666-6/UrbanSARFloods |
| STURM-Flood | 10m SAR+optical | 60 events, 24K tiles | https://zenodo.org/records/12748983 |
| Global Flood Database (MODIS) | 250m | 913 events (2000-2018) | https://global-flood-database.cloudtostreet.ai/ |

---

## 8. DDPM_REMD Code Structure & Adaptation Guide

### 8.1 Repository Layout

Clone: `git clone https://github.com/tiwarylab/DDPM_REMD.git`

```
DDPM_REMD/
├── denoising_diffusion_pytorch/
│   └── denoising_diffusion_pytorch.py   ← THE core file (611 lines, everything is here)
├── run_training.py                      ← Training driver script
├── gen_sample.py                        ← Sampling/generation script
├── traj_AIB9/                           ← Example training data (AIB9 peptide)
│   ├── AIB9_REMD_T_full_100000ps_2.0ps_traj.npy   ← (500000, 19)
│   └── sample_T_traj.npy               ← Conditioning values for generation
├── traj_GACC/                           ← Example training data (GACC RNA)
└── requirements.txt
```

### 8.2 Key Classes in `denoising_diffusion_pytorch.py`

| Class | Purpose | Lines (approx) |
|---|---|---|
| `SinusoidalPosEmb` | Timestep embedding | ~10 |
| `Block`, `ResnetBlock` | Basic conv blocks | ~30 |
| `LinearAttention`, `Attention` | Attention layers | ~30 |
| `Unet` | 1D U-Net architecture | ~100 |
| `GaussianDiffusion` | Forward/reverse diffusion, loss, sampling | ~150 |
| `Dataset_traj` | Data loading from .npy files | ~20 |
| `Trainer` | Training loop with EMA | ~100 |

### 8.3 Architecture: 1D U-Net (NOT 2D)

**Critical detail**: The U-Net uses `Conv1d`, not `Conv2d`. Data is treated as a **1D sequence**, not a 2D image.

```python
# Architecture parameters (from run_training.py)
model = Unet(
    dim=32,           # Base channel dimension
    dim_mults=(1, 2, 2, 4),  # Channel multipliers per resolution level
    groups=8,         # GroupNorm groups
    channels=1        # Input channels (FIXED at 1)
)
```

- Input shape to U-Net: `(batch_size, 1, sequence_length)` where sequence_length = number of features
- For AIB9: sequence_length = 19 (1 temperature + 18 dihedral angles)
- For our flood data: sequence_length = N_storm_params + N_surge_points

### 8.4 How Conditioning Works (Inpainting, NOT Conditional Input)

This is the most important implementation detail. Wang et al. do NOT pass θ as a separate conditional input to the U-Net. Instead, they use **inpainting**:

1. The conditioning variable (temperature T, or our storm parameter θ) is placed as **column 0** of the data vector
2. During diffusion, column 0 is **never noised** — it is masked from the noise process
3. During denoising (generation), column 0 is set to the desired conditioning value and **held fixed** throughout all denoising steps
4. The loss function only computes over the non-masked columns

```python
# In GaussianDiffusion class:
# unmask_number=1 means column 0 is kept noise-free
# During training:
noise = torch.randn_like(x_start)
noise[:, :, :unmask_number] = 0  # Don't noise the conditioning column

# During sampling:
x[:, :, :unmask_number] = conditioning_value  # Hold fixed at each step
```

**What this means for adaptation**: Your storm parameter θ goes in column 0 of the data. If θ is multi-dimensional (e.g., surge height + forward speed + track angle), you need to decide:
- **Option A (Recommended for POC)**: Use a single scalar θ (e.g., just peak surge at a reference point) → `unmask_number=1`
- **Option B**: Use multiple θ dimensions → increase `unmask_number` to match (e.g., `unmask_number=3` for 3 storm parameters)

### 8.5 Training Data Format

**Format**: NumPy `.npy` file

**Shape**: `(N_samples, N_features)` where:
- Column 0 (or columns 0 to `unmask_number-1`) = conditioning variable(s) θ
- Remaining columns = the response variables (surge at save points)

**Example** (AIB9):
```python
import numpy as np
data = np.load('traj_AIB9/AIB9_REMD_T_full_100000ps_2.0ps_traj.npy')
print(data.shape)  # (500000, 19) → 500K samples, col 0 = T, cols 1-18 = dihedrals
```

**For our flood data**: You need to produce a `.npy` of shape `(N_storms, 1 + N_points)`:
- `N_storms` ≈ 1,050 (NACCS) or 70 (CERA fallback)
- Column 0 = storm parameter θ (e.g., peak surge at the Battery reference gauge)
- Columns 1 to N_points = peak surge at each NYC-area save point

### 8.6 Key Hyperparameters (in `run_training.py`)

```python
# These are the defaults from Wang et al. — you WILL need to adjust some:
timesteps = 1000          # Diffusion steps (start with 500 for faster iteration)
train_num_steps = 2_000_000  # Total training steps (REDUCE — we have far less data)
batch_size = 128          # Adjust based on GPU memory and dataset size
lr = 1e-5                 # Learning rate (Adam)
op_num = 18               # CHANGE THIS: number of order parameters (= N_surge_points)
unmask_number = 1         # Number of conditioning columns (= dimensionality of θ)
ema_decay = 0.995         # EMA for model weights
```

**Adjustments needed for flood data**:
- `op_num`: Set to the number of NYC-area surge points you extract
- `train_num_steps`: With ~1,050 samples (vs. 500,000), reduce significantly — start with 50,000-100,000 and monitor loss
- `batch_size`: With ~1,050 samples, use 32 or 64
- `unmask_number`: 1 if using a single θ, more if using multi-dimensional θ

### 8.7 Sampling / Generation (in `gen_sample.py`)

To generate at a desired θ value:
1. Create a `.npy` file with the desired θ values in column 0 (column 0 values, rest can be zeros or random)
2. Run `gen_sample.py` which loads a trained model checkpoint and generates samples conditioned on those θ values
3. Output: generated surge values at all save points for each requested θ

```python
# Example: generate at 100 evenly spaced θ values from 1.0 to 5.0
import numpy as np
theta_values = np.linspace(1.0, 5.0, 100)
sample_input = np.zeros((100, 1 + N_points))
sample_input[:, 0] = theta_values
np.save('sample_theta.npy', sample_input)
```

### 8.8 Files You Need to Modify

| File | What to Change |
|---|---|
| `run_training.py` | `op_num`, `train_num_steps`, `batch_size`, data file path, output directory |
| `gen_sample.py` | `op_num`, model checkpoint path, sample conditioning file path |
| `denoising_diffusion_pytorch.py` | Usually **no changes needed** — the architecture is generic |

---

## 9. Compute & Python Environment

### 9.1 Hardware Requirements

**GPU is required for training.** The DDPM_REMD code uses PyTorch with CUDA.

| Option | Pros | Cons |
|---|---|---|
| **Google Colab (free tier)** | Free, T4 GPU, easy setup | 12hr session limit, may disconnect |
| **Google Colab Pro** ($10/month) | A100 GPU, longer sessions | Costs money |
| **NYU HPC (Greene cluster)** | Powerful GPUs, no cost for NYU | Requires NYU HPC account, queue wait times |
| **Local Mac (Apple Silicon)** | Already available | PyTorch MPS backend has limited support; DDPM_REMD may not work out-of-the-box on MPS |

**Recommendation**: Start with Google Colab (free). If session limits are problematic, use NYU HPC or Colab Pro.

**Note on Apple Silicon Macs**: PyTorch's MPS backend (for M1/M2/M3 GPUs) has incomplete support for some operations. The DDPM_REMD code was written for CUDA. It *may* work on MPS with minor modifications, but debugging MPS compatibility is not a good use of limited time. Use Colab instead.

### 9.2 Python Environment

```bash
# Create environment
conda create -n ddpm_flood python=3.10 -y
conda activate ddpm_flood

# Core dependencies (from DDPM_REMD)
pip install torch torchvision  # Use CUDA version if on GPU machine
pip install einops              # Tensor operations used in U-Net
pip install tqdm                # Progress bars
pip install pillow              # Image handling (used in Trainer class)
pip install numpy               # Data handling

# For data preprocessing (NACCS/CERA)
pip install geopandas           # Read geodatabases (.gdb)
pip install fiona               # Backend for geopandas geodatabase support
pip install netCDF4             # Read CERA NetCDF files
pip install xarray              # Higher-level NetCDF interface
pip install shapely             # Geometric operations (bounding box filtering)
pip install pyproj              # Coordinate transformations

# For visualization
pip install matplotlib
pip install seaborn
pip install cartopy             # Map plotting (optional, for geographic context)

# For Colab (if using):
# Most of these are pre-installed. Just need:
# !pip install einops geopandas fiona netCDF4 xarray cartopy
```

### 9.3 Colab Setup Script

```python
# Run this cell first in Colab:
!pip install einops geopandas fiona netCDF4 xarray -q
!git clone https://github.com/tiwarylab/DDPM_REMD.git
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 10. Data Preprocessing Pipeline

### 10.1 NYC Bounding Box for NACCS Save Point Extraction

The NACCS dataset has ~19,000 save points along the entire North Atlantic coast. You need to extract only the NYC-area points.

**NYC/Lower Manhattan Bounding Box** (approximate):
```python
# Bounding box for Lower Manhattan + surrounding area
NYC_BBOX = {
    'min_lon': -74.06,  # West (past Liberty Island)
    'max_lon': -73.96,  # East (past Brooklyn waterfront)
    'min_lat': 40.68,   # South (past Governors Island)
    'max_lat': 40.78    # North (past Midtown)
}

# Broader NYC metro area (if you want more save points):
NYC_METRO_BBOX = {
    'min_lon': -74.30,  # West (past Newark)
    'max_lon': -73.70,  # East (past JFK)
    'min_lat': 40.45,   # South (past Sandy Hook)
    'max_lat': 40.95    # North (past Yonkers)
}
```

**Start with the broader metro bbox** — you want enough save points for a meaningful data vector. If there are too many (hundreds), you can cluster or subsample.

### 10.2 Reading NACCS Geodatabase

```python
import geopandas as gpd
from shapely.geometry import box

# Read the NACCS geodatabase
# The .gdb may have multiple layers — list them first:
import fiona
layers = fiona.listlayers('NACCS.gdb')
print(layers)  # Identify which layer contains save points + surge data

# Read the save points layer
gdf = gpd.read_file('NACCS.gdb', layer='<appropriate_layer_name>')
print(gdf.columns)  # Inspect what fields are available
print(gdf.head())

# Filter to NYC bounding box
nyc_box = box(NYC_METRO_BBOX['min_lon'], NYC_METRO_BBOX['min_lat'],
              NYC_METRO_BBOX['max_lon'], NYC_METRO_BBOX['max_lat'])
nyc_points = gdf[gdf.geometry.within(nyc_box)]
print(f"Found {len(nyc_points)} save points in NYC area")
```

### 10.3 Reading CERA/DesignSafe NetCDF (Fallback)

```python
import xarray as xr

# Open a CERA ADCIRC output file
ds = xr.open_dataset('sandy_maxele.63.nc')  # Max elevation file
print(ds)  # See all variables
print(ds.data_vars)

# Key variables in ADCIRC output:
# - zeta_max: maximum water surface elevation at each node
# - x, y: longitude/latitude of each node
# Filter to NYC area using the same bounding box approach
```

### 10.4 Building the Training Data Matrix

The goal is to produce a single `.npy` file that the DDPM_REMD code can consume:

```python
import numpy as np

# After extracting NYC save points and storm parameters:
# storm_params: array of shape (N_storms,) — the conditioning variable θ
# surge_matrix: array of shape (N_storms, N_points) — surge at each point per storm

# Combine into DDPM input format: column 0 = θ, columns 1+ = surge
training_data = np.column_stack([storm_params.reshape(-1, 1), surge_matrix])
print(f"Training data shape: {training_data.shape}")  # Should be (N_storms, 1 + N_points)

# Normalize the data (important for DDPM training stability)
# Option 1: Min-max normalize each column to [0, 1]
# Option 2: Standardize (zero mean, unit variance) — closer to what Wang et al. likely did
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
training_data_normalized = scaler.fit_transform(training_data)

# Save
np.save('flood_training_data.npy', training_data_normalized)

# Also save the scaler for inverse-transforming generated samples:
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### 10.5 Storm Parameter θ — What to Use

The NACCS storms are parameterized by multiple variables. For the conditioning parameter θ:

| Approach | θ Dimensionality | `unmask_number` | Pros | Cons |
|---|---|---|---|---|
| **Peak surge at Battery (scalar)** | 1D | 1 | Simplest, direct analog to Wang's T | Loses info about storm shape |
| **Storm intensity + forward speed** | 2D | 2 | Captures more physics | Slightly more complex |
| **Intensity + speed + track angle** | 3D | 3 | Most physical | May need more data |

**Recommendation**: Start with **1D (peak surge at a reference point)** for the proof-of-concept. This is the closest analog to Wang et al.'s temperature T and requires zero code changes to `unmask_number`.

### 10.6 Note: No Direct Boltzmann Weight Analog

In Wang et al., generated molecular configurations carry **Boltzmann weights** that allow discarding unphysical hallucinations (configurations with negligible thermodynamic probability).

For flood data, there is **no direct analog** to Boltzmann weights. This means:
- You cannot use the same automatic plausibility filtering
- Instead, validate generated samples using: (a) physical bounds (surge ≥ 0, surge ≤ reasonable max), (b) spatial consistency (nearby points should have similar surge), (c) comparison to ADCIRC held-out set
- This is an honest limitation to acknowledge in the presentation — and a good "future work" direction (e.g., "develop physics-informed plausibility metrics for generated flood scenarios")

---

## 11. Execution Plan (7 Days)

### Day 1: Environment Setup + Data Acquisition
- [ ] Set up compute environment (see Section 9):
  - **Recommended**: Google Colab — run the Colab Setup Script from Section 9.3
  - **Alternative**: NYU HPC (if you have access) or local with CUDA GPU
- [ ] Install Python dependencies (see Section 9.2)
- [ ] Clone DDPM_REMD repo: `git clone https://github.com/tiwarylab/DDPM_REMD.git`
- [ ] Clone GISSR repo: `git clone https://github.com/ym2540/GIS_FloodSimulation.git`
- [ ] **Run the AIB9 example first** to verify the code works:
  ```bash
  cd DDPM_REMD
  python run_training.py  # Should start training on AIB9 data
  # Let it run for ~1000 steps to confirm no errors, then cancel
  ```
- [ ] Register for DesignSafe-CI (free): https://www.designsafe-ci.org/
- [ ] Download NACCS geodatabase from NROC: https://www.northeastoceancouncil.org/naccs/
- [ ] Download CERA Sandy hindcast from PRJ-3932 (NetCDF) as backup
- [ ] Download Sandy Inundation Zone from NYC Open Data
- **Fallback**: If NACCS access is slow, pivot to CERA 70-storm dataset entirely

### Day 2: Data Exploration + Preprocessing
- [ ] Explore NACCS geodatabase structure (see Section 10.2 for code):
  - List all layers with `fiona.listlayers()`
  - Identify which layer has save points + surge values
  - Identify which fields contain storm parameters
- [ ] Filter save points to NYC metro bounding box (see Section 10.1):
  - Start with broader bbox (min_lon=-74.30 to max_lon=-73.70, min_lat=40.45 to max_lat=40.95)
  - Record how many save points fall within the bbox
- [ ] Extract for each storm: storm parameters (θ) + peak surge at NYC-area save points
- [ ] Verify Sandy storm data matches known values (~3.4m surge at Battery)
- [ ] Build training data `.npy` (see Section 10.4):
  - Shape: `(N_storms, 1 + N_points)` — column 0 = θ, rest = surge values
  - Normalize with StandardScaler
  - Save scaler object for inverse transform
- **Output**: `flood_training_data.npy` + `scaler.pkl`

### Days 3-4: DDPM Code Adaptation + Training
- [ ] Read Section 8 of this document thoroughly (code structure + adaptation guide)
- [ ] Adapt `run_training.py`:
  - Set `op_num` = N_surge_points (number of NYC save points you extracted)
  - Set `unmask_number` = 1 (conditioning on scalar θ; see Section 10.5)
  - Set data file path to your `flood_training_data.npy`
  - Reduce `train_num_steps` to 50,000–100,000 (we have ~1,050 samples, not 500,000)
  - Reduce `batch_size` to 32 or 64
  - Start with `timesteps=500` (faster training; increase to 1000 if quality is insufficient)
- [ ] **No changes needed** in `denoising_diffusion_pytorch.py` — the architecture is generic
- [ ] Train DDPM on full dataset (hold out ~15-20% for validation)
- [ ] Monitor training loss convergence — should converge in hours, not days
- **Note**: If loss plateaus early, try reducing `dim` from 32 to 16 (our data is lower-dimensional)

### Days 5-6: Experiments + Figure Generation
- [ ] **Experiment 1 — Interpolation**: Generate at held-out storm parameters, compare DDPM vs. ADCIRC ground truth
  - Metric: R² and RMSE between DDPM-generated and ADCIRC-actual surge at held-out storms
  - Output: scatter plot (see Section 12, Figure 1)
- [ ] **Experiment 2 — Extrapolation**: Retrain on storms with peak surge ≤ median only, then generate above median, compare to ADCIRC
  - Metric: RMSE in the extrapolated region vs. interpolated region
  - Output: line plot with extrapolation boundary (see Section 12, Figure 2)
- [ ] **Experiment 3 — Tipping-point discovery**: Use `gen_sample.py` to generate at 100-200 evenly spaced θ values spanning the full range
  - Plot surge at each NYC location vs. θ → look for sharp jumps (discontinuities in the response curve)
  - Tipping point = θ value where d(surge)/d(θ) exceeds a threshold (e.g., 2× the mean gradient)
  - Output: response curves (see Section 12, Figure 3)
- [ ] **Validation**: Compare Sandy scenario — find the NACCS storm closest to Sandy's parameters, compare DDPM prediction vs. actual Sandy observations (USGS high water marks)
- [ ] If time permits: compare with GISSR predictions (Tables 5-7 from Miura et al. 2021)
- [ ] **Plausibility checks** (since no Boltzmann weight analog — see Section 10.6):
  - Verify generated surge values ≥ 0
  - Verify spatial consistency (nearby save points have similar surge)
  - Flag any generated samples with extreme outlier values
- **Output**: 3-4 publication-quality figures

### Day 7: Presentation Assembly
- [ ] Build 12-15 slides (see Section 10 for structure)
- [ ] Write speaker notes
- [ ] Prepare for potential questions from Prof. Miura

---

## 12. Expected Outputs & Figures

### Figure 1: Interpolation Accuracy (Scatter Plot)
```
X-axis: ADCIRC peak surge at NYC points (ground truth, held-out storms)
Y-axis: DDPM-generated peak surge at same points
Each dot = one (storm, location) pair
Perfect prediction = dots on diagonal line
Include R² score and RMSE
```
**Proves**: DDPM accurately generates flood response at storm parameters it never saw during training.

### Figure 2: Extrapolation Beyond Training Range (Line Plot)
```
X-axis: Storm intensity parameter θ (e.g., peak surge from 1m to 5m)
Y-axis: Predicted flood depth at a critical location (e.g., Battery Park)
Blue line: ADCIRC ground truth
Red line: DDPM predictions
Vertical dashed line: boundary of training data
Right of line = DDPM extrapolating to untested extremes
```
**Proves**: DDPM can predict flood behavior for storms more extreme than anything in the training set.

### Figure 3: Tipping-Point Discovery (Response Curves) — THE KEY NOVEL FIGURE
```
X-axis: Storm parameter θ (surge height, continuously varied)
Y-axis: DDPM-generated flood depth
Multiple curves for different Lower Manhattan locations
Most show gradual increase
Some show SHARP JUMP at specific θ → tipping point
Annotate: "At θ = X.Xm, Location Y transitions from dry to catastrophic"
```
**Proves**: DDPM discovers critical flood thresholds automatically — the "transition states" of urban flooding.

### Figure 4: Comparison Table (DDPM vs. ADCIRC vs. GISSR)
```
                | ADCIRC       | DDPM              | GISSR
Accuracy        | Baseline     | ~95%+ of ADCIRC   | ~83% (from paper)
Speed/sim       | Hours        | < 1 second         | 0.01 second
Extrapolation   | No (new run) | Yes                | Yes but simplified
Distributional  | No           | Yes                | No
Tipping points  | Not efficient| Yes                | Not efficient
```
**Proves**: DDPM bridges the gap between ADCIRC fidelity and GISSR speed.

---

## 13. Presentation Structure

### Slide 1: Title
"Discovering Flood Tipping Points with Denoising Diffusion Probabilistic Models"

### Slides 2-3: Motivation
- Climate change + SLR → increasing flood risk in coastal cities
- Current tools: ADCIRC (accurate but slow), GISSR (fast but simplified)
- Gap: No tool efficiently discovers critical flood thresholds across the storm parameter space

### Slides 4-5: Wang et al. Framework
- DDPM for mixing physics across temperatures
- Key insight: treat control parameter as random variable
- Demonstrated: discovers transition states, extrapolates beyond training range
- The 5 unique strengths of DDPM

### Slides 6-7: Conceptual Mapping to Floods
- The mapping table (Temperature → Storm parameters, Configurations → Flood response, etc.)
- Why Aurora doesn't compete (resolution, scope, infrastructure modeling)

### Slides 8-9: Data & Methodology
- NACCS ADCIRC: 1,050 synthetic storms, ~19,000 save points
- Preprocessing: extract NYC subdomain, pair with storm parameters
- DDPM architecture adaptation from Wang et al.

### Slides 10-13: Results (The Core)
- Figure 1: Interpolation accuracy
- Figure 2: Extrapolation capability
- Figure 3: Tipping-point discovery (THE headline result)
- Figure 4: Comparison with ADCIRC and GISSR

### Slide 14: Discussion & Novelty
- DDPM bridges the ADCIRC-GISSR gap
- Enables probabilistic flood risk assessment
- Discovers critical thresholds no single tool can find efficiently

### Slide 15: Future Directions
- Multi-fidelity training (ADCIRC + GISSR + real observations)
- Extension to multiple cities
- Integration with protective strategy optimization (connects back to Miura et al. 2021's optimization goal)
- Full 2D flood map generation with higher-fidelity training data

---

## 14. References & External Links

### Papers (Local Copies)
1. **Wang et al. 2022** — "From Data to Noise to Data for Mixing Physics Across Temperatures with Generative Artificial Intelligence" — PNAS 119(32)
   - Local: `wang-et-al-2022-from-data-to-noise-to-data-for-mixing-physics-across-temperatures-with-generative-artificial (1).pdf`
   - DOI: https://doi.org/10.1073/pnas.2203656119

2. **Miura et al. 2021** — "High-Speed GIS-Based Simulation of Storm Surge–Induced Flooding Accounting for Sea Level Rise" — Natural Hazards Review 22(3)
   - Local: `miura-et-al-2021-high-speed-gis-based-simulation-of-storm-surge-induced-flooding-accounting-for-sea-level-rise (1).pdf`
   - DOI: https://doi.org/10.1061/(ASCE)NH.1527-6996.0000465

### Code Repositories
- **DDPM_REMD** (Wang et al.): https://github.com/tiwarylab/DDPM_REMD
- **GISSR** (Miura et al.): https://github.com/ym2540/GIS_FloodSimulation

### Key Data Portals
- **NACCS Data**: https://www.northeastoceancouncil.org/naccs/
- **DesignSafe CERA/ADCIRC**: https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3932
- **CERA Historical Storms**: https://historicalstorms.coastalrisk.live/
- **NYC Open Data**: https://data.cityofnewyork.us/
- **NOAA Tides & Currents**: https://tidesandcurrents.noaa.gov/
- **USGS Flood Event Viewer**: https://stn.wim.usgs.gov/FEV/
- **NYC FloodNet**: https://www.floodnet.nyc/

### Prof. Miura's Lab
- **CERA Lab**: https://yukimiura.org/
- **NYU Faculty Profile**: https://engineering.nyu.edu/faculty/yuki-miura
- **GISSR GitHub**: https://github.com/ym2540/GIS_FloodSimulation

### Microsoft Aurora
- **Article**: https://news.microsoft.com/source/features/ai/microsofts-aurora-ai-foundation-model-goes-beyond-weather-forecasting/

---

## Appendix: Fallback Data Strategy

If NACCS data proves difficult to access or process in time:

| Priority | Source | # Scenarios | Effort |
|---|---|---|---|
| Plan A | NACCS geodatabase (1,050 storms) | Best coverage | Medium (GIS processing) |
| Plan B | CERA/DesignSafe (70 storms) | Sufficient for proof-of-concept | Low (NetCDF/GeoTIFF) |
| Plan C | CERA GeoTIFF max inundation maps | Simplest format | Very low |

The minimum viable dataset is the CERA 70-storm archive — 70 storms with full ADCIRC outputs is enough to demonstrate all three key experiments (interpolation, extrapolation, tipping points) for a proof-of-concept presentation.
