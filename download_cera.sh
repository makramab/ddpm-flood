#!/bin/bash
# Download CERA maxele.63.nc files directly from TACC
# Uses SSH multiplexing so you only enter password/token ONCE

REMOTE="makramab@data.tacc.utexas.edu"
BASE="/corral/projects/NHERI/published/published-data/PRJ-3932/Simulation--adcircswan-hindcasts-for-historical-storms-2003-2023--V3/data/Model--adcirc-readme/Input--adcirc-inputs/Output--adcirc-outputs/data/Outputs"
LOCAL="data/cera"

mkdir -p "$LOCAL"

# Set up SSH multiplexing (enter password/token once, reuse for all downloads)
SOCK="/tmp/tacc_ssh_sock"
echo "=== Setting up SSH connection (enter password + token once) ==="
ssh -fNM -S "$SOCK" "$REMOTE"

if [ $? -ne 0 ]; then
    echo "SSH connection failed. Exiting."
    exit 1
fi

echo "=== SSH connection established. Downloading files... ==="

# List of storms: YEAR/NUM_NAME/MESH_FOLDER
declare -a STORMS=(
    "2003/13_ISABEL/STOFSatl_al132003_gahm_swan"
    "2004/01_ALEX/SABv20a_al012004_gahm_swan"
    "2004/03_CHARLEY/STOFSatl_al032004_gahm_swan"
    "2004/06_FRANCES/HSOFS_al062004_gahm_swan"
    "2004/07_GASTON/SABv20a_al072004_gahm_swan"
    "2004/11_JEANNE/SABv20a_al112004_gahm_swan"
    "2005/16_OPHELIA/SABv20a_al162005_gahm_swan"
    "2008/15_OMAR/HSOFS_al152008_gahm_swan"
    "2010/07_EARL/HSOFS_al072010_gahm_swan"
    "2011/09_IRENE/HSOFS_al092011_gham_swam"
    "2012/04_DEBBY/HSOFS_al042012_gahm_swan"
    "2012/18_SANDY/HSOFS_al182012_gahm_swan"
    "2013/01_ANDREA/HSOFS_al012013_gahm_swan"
    "2014/01_ARTHUR/HSOFS_al012014_gahm_swan"
    "2016/14_MATTHEW/SABv20a_al142016_gahm_Swan"
    "2017/15_MARIA/HSOFS_al152027_gahm_swan"
    "2018/01_ALBERTO/HSOFS_al012018_gahm_swan"
    "2018/06_FLORENCE/SABv20a_al062018_gahm_swan"
    "2018/14_MICHAEL/HSOFS_al142018_gahm_swan"
    "2019/05_DORIAN/SABv20a_al052019_gahm_swan"
    "2020/06_FAY/STOFSatl_al062020_gahm_swan"
    "2020/09_ISAIAS/SABv20a_al092020_gahm_swan"
    "2020/29_ETA/STOFSatl_al292020_gahm_swan"
    "2021/08_HENRI/NAC2014_al082021_gahm_swan"
    "2022/07_FIONA/HSOFS_al072022_gahm_swan"
    "2022/09_IAN/STOFSatl_al092022_gahm_swan"
    "2022/17_NICOLE/HSOFS_al172022_qb_gahm_swan"
    "2023/10_IDALIA/STOFSatl_al102023_gahm_swan"
    "2023/16_OPHELIA/SABv20a_al162023_gahm_swan"
    "2024/02_BERYL/STOFSatl_al022024_gahm_swan"
    "2024/22_DEBBY/STOFSatl_al022024_gahm_swan"
)

TOTAL=${#STORMS[@]}
COUNT=0
SKIPPED=0

for entry in "${STORMS[@]}"; do
    COUNT=$((COUNT + 1))
    # Extract a clean name: 2003/13_ISABEL/... -> 2003_13_ISABEL
    clean_name=$(echo "$entry" | cut -d'/' -f1,2 | tr '/' '_')
    dest_file="${LOCAL}/${clean_name}_maxele.63.nc"

    # Skip if already downloaded
    if [ -f "$dest_file" ]; then
        echo "[$COUNT/$TOTAL] SKIP (exists): $clean_name"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$COUNT/$TOTAL] Downloading: $clean_name ..."
    scp -o "ControlPath=$SOCK" "${REMOTE}:${BASE}/${entry}/maxele.63.nc" "$dest_file"

    if [ $? -eq 0 ]; then
        size=$(du -h "$dest_file" | cut -f1)
        echo "         Done: $size"
    else
        echo "         FAILED"
    fi
done

# Close SSH connection
ssh -S "$SOCK" -O exit "$REMOTE" 2>/dev/null

echo ""
echo "=== Download complete ==="
echo "Files downloaded: $((COUNT - SKIPPED))"
echo "Files skipped (already existed): $SKIPPED"
echo "Total files in $LOCAL:"
ls "$LOCAL"/*.nc 2>/dev/null | wc -l
du -sh "$LOCAL"
