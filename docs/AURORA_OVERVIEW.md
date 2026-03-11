# Deep Comparison: Conditional DDPM Flood Generation vs Microsoft Aurora Foundation Model

Author: Prepared for expert review and technical discussion.

---

# 1. Overview

This document summarizes a deep comparison between the current *conditional DDPM-based coastal flood generation approach* and *Microsoft Aurora*, an Earth-system foundation model introduced in Nature (2025).

Aurora represents a new class of *AI-based Earth-system foundation models* capable of forecasting atmospheric and oceanic states at scale. The goal of this document is to position Aurora as a *future research direction or complementary modeling approach* relative to the current diffusion-based hazard generation model.

This comparison aims to remain grounded in the *current DDPM experimental results*, while exploring how Aurora could serve as a potential upstream or alternative modeling strategy.

---

# 2. Current Work: Conditional DDPM for Coastal Flood Map Generation

The current research uses a *conditional Denoising Diffusion Probabilistic Model (DDPM)* to generate spatial flood or storm surge maps.

Conceptually the model learns the conditional distribution:

p(flood_map | scenario_parameters)

Where:

*flood_map*

Spatial hazard representation such as:

•⁠  ⁠storm surge height field
•⁠  ⁠coastal inundation raster
•⁠  ⁠flood probability grid

Typical resolution examples:

•⁠  ⁠64 × 64 spatial grid
•⁠  ⁠coastal surge intensity field

*scenario_parameters*

Environmental conditioning variables such as:

•⁠  ⁠Sea Level Rise (SLR)
•⁠  ⁠Storm surge level
•⁠  ⁠Tide level
•⁠  ⁠Storm characteristics (track, wind speed, pressure)
•⁠  ⁠Possibly environmental measurements or gauges

---

## 2.1 Key Characteristics of the DDPM Approach

The diffusion model learns the *distribution of possible hazard outcomes* conditioned on environmental parameters.

Important properties:

•⁠  ⁠Explicit *generative modeling*
•⁠  ⁠Produces multiple possible hazard realizations
•⁠  ⁠Can capture *uncertainty and variability*
•⁠  ⁠Allows *scenario exploration*

Example use cases:

•⁠  ⁠Extreme event generation
•⁠  ⁠Risk simulation
•⁠  ⁠Climate adaptation planning
•⁠  ⁠Coastal resilience analysis

Unlike deterministic forecasting models, the diffusion model explicitly models the *probability distribution of spatial hazard fields*.

---

# 3. Microsoft Aurora: Earth-System Foundation Model

Microsoft Aurora is a *1.3 billion parameter Earth-system foundation model* trained on more than *one million hours of geophysical data*.

Nature publication:

https://www.nature.com/articles/s41586-025-09005-y

GitHub repository:

https://github.com/microsoft/aurora

Aurora is designed as a *general-purpose geophysical forecasting model* capable of predicting multiple Earth-system variables.

---

## 3.1 Aurora Architecture

Aurora follows an:

Encoder → Processor → Decoder architecture.

Components:

*Encoder*

Perceiver-style encoder that ingests heterogeneous geophysical inputs.

*Processor*

A *3D Swin Transformer* architecture that models spatiotemporal Earth-system dynamics.

*Decoder*

Produces predictions of physical variables such as weather or ocean conditions.

---

## 3.2 Aurora Prediction Tasks

Aurora can be fine-tuned for several forecasting tasks including:

•⁠  ⁠Medium-resolution weather forecasting
•⁠  ⁠High-resolution weather prediction
•⁠  ⁠Tropical cyclone track prediction
•⁠  ⁠Ocean wave prediction
•⁠  ⁠Air pollution forecasting

The model focuses on predicting *Earth-system state variables* rather than downstream hazard outcomes.

Examples of predicted variables include:

•⁠  ⁠wind velocity
•⁠  ⁠atmospheric pressure
•⁠  ⁠temperature
•⁠  ⁠ocean wave fields
•⁠  ⁠cyclone trajectories

---

# 4. Architectural Comparison

## 4.1 DDPM Approach

Key properties:

•⁠  ⁠Diffusion-based generative model
•⁠  ⁠Explicitly models probability distributions
•⁠  ⁠Scenario-conditioned hazard generation
•⁠  ⁠Produces spatial flood / surge maps directly

Output target:

Coastal hazard fields.

Examples:

•⁠  ⁠flood raster
•⁠  ⁠surge height grid
•⁠  ⁠inundation probability map

Learning objective:

Model the conditional distribution:

p(hazard_map | environmental_conditions)

---

## 4.2 Aurora Approach

Key properties:

•⁠  ⁠Transformer-based Earth-system model
•⁠  ⁠Foundation model trained on large geophysical datasets
•⁠  ⁠Predictive rather than generative hazard modeling
•⁠  ⁠Focus on atmospheric and ocean dynamics

Output target:

Earth-system state variables.

Examples:

•⁠  ⁠wind
•⁠  ⁠pressure
•⁠  ⁠temperature
•⁠  ⁠waves
•⁠  ⁠cyclone tracks

Learning objective:

Predict future physical states of the Earth system.

---

# 5. Core Conceptual Difference

The most important difference between the two approaches is the *layer of the modeling pipeline where they operate*.

*DDPM*

Operates at the *hazard generation layer*.

The model directly generates spatial flood or surge outcomes.

*Aurora*

Operates at the *Earth-system forecasting layer*.

The model predicts atmospheric and oceanic conditions.

---

## 5.1 Modeling Pipeline Interpretation

Typical Earth-system modeling pipeline:

Weather model → Ocean model → Surge model → Flood model

Aurora sits at the *weather / ocean state forecasting stage*.

The DDPM approach sits at the *hazard generation stage*.

Simplified comparison:

Aurora → environmental conditions → surge / flood model

DDPM → directly generate surge / flood hazard maps

Thus Aurora is not a direct replacement for the DDPM approach but instead represents a *potential upstream modeling component*.

---

# 6. Potential Hybrid Architecture

A future modeling pipeline could combine both approaches.

Example hybrid workflow:

1.⁠ ⁠Aurora generates large-scale environmental conditions.

Example outputs:

•⁠  ⁠storm tracks
•⁠  ⁠wind fields
•⁠  ⁠atmospheric pressure
•⁠  ⁠ocean wave conditions

2.⁠ ⁠These variables serve as *conditioning inputs* for the DDPM model.

3.⁠ ⁠The DDPM generates spatial flood or surge hazard maps.

Example hybrid pipeline:

Aurora → environmental forcing → DDPM → flood map generation

---

## 6.1 Potential Benefits of Hybrid Approach

Advantages:

Aurora captures:

•⁠  ⁠large-scale atmospheric physics
•⁠  ⁠global weather dynamics
•⁠  ⁠cyclone development

DDPM captures:

•⁠  ⁠localized hazard distributions
•⁠  ⁠uncertainty in flood outcomes
•⁠  ⁠scenario-based coastal impacts

This combination could produce *physically consistent yet generative coastal hazard simulations*.

---

# 7. Citation Landscape Around Aurora

Aurora has quickly accumulated a growing citation network.

Google Scholar citation page:

https://scholar.google.com/scholar?oi=bibs&hl=en&cites=14983679548429028479

Many citing works focus on:

•⁠  ⁠AI weather forecasting
•⁠  ⁠tropical cyclone prediction
•⁠  ⁠extreme event analysis
•⁠  ⁠benchmarking AI weather models

Only a small subset of research appears closely aligned with *coastal hazard or storm surge applications*.

---

## 7.1 Example Relevant Citations

### AI-generated tropical cyclone benchmarking

https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JD044753

This work evaluates AI weather models including Aurora in the context of tropical cyclone dynamics and discusses implications for storm surge conditions.

---

### Coastal resilience optimization with AI prediction

https://arxiv.org/html/2509.19263v1

This work studies coastal resilience planning and references Aurora as a frontier Earth-system forecasting model relevant to surge prediction.

---

### Observational evaluation of AI weather models

https://arxiv.org/html/2509.01879v1

This paper benchmarks several AI weather models including Aurora, focusing on forecasting performance.

---

### Extreme event predictability using AI weather models

https://arxiv.org/pdf/2603.06516

This work examines the predictability of extreme weather events using AI-based forecasting models including Aurora.

---

# 8. Key Insight

Based on current literature:

Very few studies currently adapt Aurora directly for *coastal flood map generation*.

Most Aurora-related work focuses on:

•⁠  ⁠weather forecasting
•⁠  ⁠cyclone dynamics
•⁠  ⁠atmospheric modeling
•⁠  ⁠extreme event prediction

Therefore the current DDPM approach remains valuable because it directly targets *hazard-field generation*, which is still relatively underexplored compared with general Earth-system forecasting.

---

# 9. Future Research Directions

Aurora motivates several possible future research directions.

Possible avenues include:

1.⁠ ⁠Using Aurora outputs as conditioning inputs for diffusion-based flood models.

2.⁠ ⁠Fine-tuning Aurora specifically for storm surge prediction tasks.

3.⁠ ⁠Integrating Aurora outputs with physical surge simulators such as ADCIRC.

4.⁠ ⁠Developing hybrid foundation-model + generative-hazard architectures.

This positions the DDPM approach not as a competing method but as a *complementary hazard-generation layer built on top of Earth-system foundation models*.

---

# 10. References

Microsoft Aurora Nature Paper

https://www.nature.com/articles/s41586-025-09005-y

Aurora GitHub Repository

https://github.com/microsoft/aurora

Google Scholar citation page

https://scholar.google.com/scholar?oi=bibs&hl=en&cites=14983679548429028479

Climatological benchmarking of AI-generated tropical cyclones

https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JD044753

Discovering strategies for coastal resilience with AI-based prediction and optimization

https://arxiv.org/html/2509.19263v1

Observations-focused assessment of global AI weather prediction models

https://arxiv.org/html/2509.01879v1

Evaluating predictability of weather extremes using AI models

https://arxiv.org/pdf/2603.06516