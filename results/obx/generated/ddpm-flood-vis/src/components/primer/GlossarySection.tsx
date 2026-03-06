import { Card, CardContent } from '#/components/ui/card'
import { Section, B } from './shared'

const GLOSSARY_ENTRIES = [
  { term: 'ADCIRC', definition: 'Advanced Circulation Model — a physics-based computer program that simulates how ocean water moves under the influence of wind, atmospheric pressure, and tides. It solves the shallow water equations on an unstructured triangular mesh. Used by NOAA, FEMA, and the US Army Corps of Engineers for official storm surge forecasting.' },
  { term: 'Bathymetry', definition: 'The measurement of water depth. In ADCIRC, the depth variable at each node represents the distance below sea level (positive values) or elevation above sea level (negative values). Bathymetry governs how surge propagates — shallow water amplifies surge through a process called shoaling.' },
  { term: 'CERA', definition: 'Coastal Emergency Risks Assessment — a real-time storm surge guidance system that runs ADCIRC simulations during active hurricanes using actual meteorological observations. CERA archives provide "ground truth" simulations of historical hurricanes with realistic (non-parametric) wind forcing.' },
  { term: 'Conditioning variable', definition: 'An input to a generative model that controls what is generated. In this project, the conditioning variables are theta, latitude, longitude, and depth. They are held fixed during generation, and the model produces output consistent with these values.' },
  { term: 'Danso et al. 2025', definition: 'A published dataset of 180 ADCIRC simulations (20 hurricanes x 3 sea level scenarios x 3 wind forcing types) available on Zenodo. Uses the Holland parametric wind model for forcing.' },
  { term: 'DDPM', definition: 'Denoising Diffusion Probabilistic Model — a type of generative AI that learns to produce realistic data by training to reverse a gradual noising process. It generates samples by starting from random noise and iteratively denoising over 1,000 steps.' },
  { term: 'EMA', definition: 'Exponential Moving Average — a technique where a smoothed copy of the model weights is maintained during training by averaging recent weight updates. The EMA model is more stable and typically produces better results than the raw training model.' },
  { term: 'fort.63.nc', definition: 'An ADCIRC output file containing the full time series of water surface elevation at every mesh node at every output timestep. Much larger than maxele.63.nc but contains the complete temporal evolution of the storm.' },
  { term: 'Holland parametric model', definition: 'A mathematical formula that approximates a hurricane\'s wind field from a few parameters (central pressure deficit, radius of maximum winds, etc.). It produces idealized, roughly symmetric wind patterns. Simpler than real observed wind data, but less realistic.' },
  { term: 'HSOFS', definition: 'Hurricane Surge On-Demand Forecast System — a specific ADCIRC mesh covering the US Atlantic and Gulf coasts with approximately 1,813,443 nodes. Variable resolution: dense near coastlines (~100m spacing), sparse in open ocean (~10km spacing).' },
  { term: 'Inpainting', definition: 'A conditioning technique for diffusion models where certain columns of data are designated as "known" and never corrupted with noise during training. During generation, these columns are set to desired values and held fixed while the model generates the remaining columns.' },
  { term: 'KMeans clustering', definition: 'A standard algorithm that groups data points into K clusters based on proximity. In this project, it groups ~39,500 ADCIRC nodes into 1,974 patches of 20 geographically nearby nodes.' },
  { term: 'maxele.63.nc', definition: 'An ADCIRC output file containing the maximum (peak) water surface elevation at every mesh node over the entire simulation duration. One value per node — the highest surge that occurred at any point during the storm.' },
  { term: 'MPS', definition: 'Metal Performance Shaders — Apple Silicon\'s GPU computing framework. Used for local development and testing on Mac. Produces identical results to CUDA (NVIDIA GPU) but approximately 6x slower for this model.' },
  { term: 'NetCDF', definition: 'Network Common Data Form — a standard file format for storing scientific array data. ADCIRC output files use NetCDF format (.nc). Read using the netCDF4 Python library.' },
  { term: 'Node', definition: 'A single point (vertex) in the ADCIRC triangular mesh. Each node has coordinates (latitude, longitude), a depth value, and computed surge values. The HSOFS mesh has ~1.8 million nodes.' },
  { term: 'Patch', definition: 'A cluster of 20 geographically nearby ADCIRC nodes, created by KMeans clustering. The DDPM processes one patch at a time, generating 20 surge values conditioned on the patch\'s location. The OBX region has 1,974 patches.' },
  { term: 'R2', definition: 'Coefficient of determination — a metric measuring how well predictions match observed values. R2 = 1.0 means perfect prediction. R2 = 0.0 means predictions are no better than always predicting the mean. R2 < 0 means predictions are actively worse than the mean.' },
  { term: 'Reverse diffusion', definition: 'The generation process of a DDPM: starting from pure random noise and iteratively denoising over 1,000 steps to produce a realistic data sample. Each step slightly reduces the noise using the trained U-Net.' },
  { term: 'RMSE', definition: 'Root Mean Squared Error — the square root of the average squared difference between predictions and ground truth. Measured in the same units as the data (meters). Penalizes large errors more than small ones.' },
  { term: 'Storm surge', definition: 'The abnormal rise in sea level caused by a hurricane\'s winds pushing ocean water onshore and its low atmospheric pressure allowing the water surface to rise. Distinct from normal waves and tides. Measured in meters above normal sea level.' },
  { term: 'TACC', definition: 'Texas Advanced Computing Center — a supercomputing facility at UT Austin. Hosts the CERA archive of ADCIRC simulation outputs, including the fort.63.nc time-series files needed for the planned next step.' },
  { term: 'Theta', definition: 'The conditioning variable: peak water surface elevation at the reference tide gauge, measured in meters above mean sea level. For the Outer Banks, the reference area is near Cape Hatteras. Theta serves as a scalar severity index — a single number summarizing storm intensity for the region.' },
  { term: 'U-Net', definition: 'A neural network architecture with an encoder (compresses input), decoder (reconstructs output), and skip connections (preserve fine details). Originally designed for image segmentation, adapted here for 1D sequence data. The "U" shape comes from the encoder going down and decoder going back up.' },
  { term: 'UNMASK_NUMBER', definition: 'A parameter in the DDPM code that specifies how many columns (starting from column 0) are conditioning columns — never corrupted with noise during training and held fixed during generation. Set to 4 in this project (theta, lat, lon, depth).' },
]

export function GlossarySection() {
  return (
    <Section id="glossary" title="11. Glossary">
      <div className="grid grid-cols-1 gap-2">
        {GLOSSARY_ENTRIES.map((entry) => (
          <Card key={entry.term}>
            <CardContent className="p-3 md:p-4 flex flex-col sm:flex-row gap-1 sm:gap-4">
              <span className="shrink-0 sm:w-40">
                <B>{entry.term}</B>
              </span>
              <span className="text-sm text-muted-foreground leading-relaxed">{entry.definition}</span>
            </CardContent>
          </Card>
        ))}
      </div>
    </Section>
  )
}
