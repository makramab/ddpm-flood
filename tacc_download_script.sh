#!/bin/bash
OUTDIR=/corral/projects/NHERI/published/published-data/PRJ-3932/Simulation--adcircswan-hindcasts-for-historical-storms-2003-2023--V3/data/Model--adcirc-readme/Input--adcirc-inputs/Output--adcirc-outputs/data/Outputs
cd ~/cera_maxele
for storm_dir in "$OUTDIR"/????/*/; do
  mesh_dir=$(ls "$storm_dir" 2>/dev/null | grep -iE "HSOFS|STOFSatl|NAC2014|SABv20a")
  if [ -n "$mesh_dir" ] && [ -f "$storm_dir/$mesh_dir/maxele.63.nc" ]; then
    year_storm=$(echo "$storm_dir" | grep -oP '\d{4}/\d+_\w+')
    safe_name=$(echo "$year_storm" | tr '/' '_')
    cp "$storm_dir/$mesh_dir/maxele.63.nc" "${safe_name}_maxele.63.nc"
    echo "Copied: ${safe_name} (${mesh_dir})"
  fi
done
echo "---"
ls *.nc | wc -l
du -sh .
tar czf ~/cera_maxele_all.tar.gz *.nc
ls -lh ~/cera_maxele_all.tar.gz
