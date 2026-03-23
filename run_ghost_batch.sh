#!/bin/bash

# Configuration
INPUT_DIR="inputs"
LOG_DIR="logs"
EXE_DIR="/home/ae4872/EXE"
BASE_DIR=$(pwd)

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "================================================"
echo " Starting GHOST Batch Processing Pipeline"
echo "================================================"

# Move into the inputs directory so all output files stay there
cd "$INPUT_DIR" || exit

# Loop through all SPPLEO input files
for spp_file in *_2F_SPPLEO.inp; do
    
    # Ensure files exist
    [ -e "$spp_file" ] || { echo "No SPPLEO files found!"; break; }

    # Extract the base prefix (e.g., CMP_05_091)
    prefix="${spp_file%_2F_SPPLEO.inp}"

    echo "------------------------------------------------"
    echo "Processing Day: $prefix"
    echo "------------------------------------------------"

    # 1. Run SPPLEO
    echo "  [1/3] Running SPPLEO..."
    "$EXE_DIR/SPPLEO" "${prefix}_2F_SPPLEO.inp" > "$BASE_DIR/$LOG_DIR/${prefix}_SPPLEO.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "  [!] SPPLEO failed. Skipping $prefix."
        continue
    fi

    # 2. Run PosFit
    echo "  [2/3] Running PosFit..."
    "$EXE_DIR/PosFit" "${prefix}_2F_PosFit.inp" > "$BASE_DIR/$LOG_DIR/${prefix}_PosFit.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "  [!] PosFit failed. Skipping $prefix."
        continue
    fi

    # 3. Run ODCP
    echo "  [3/3] Running ODCP..."
    "$EXE_DIR/ODCP" "${prefix}_2F_ODCP.inp" > "$BASE_DIR/$LOG_DIR/${prefix}_ODCP.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "  [!] ODCP failed."
        continue
    fi

    echo "  [SUCCESS] $prefix completed."
done

# Return to original directory
cd "$BASE_DIR"

echo "================================================"
echo " Batch processing finished! Outputs are in ./$INPUT_DIR/"
echo "================================================"