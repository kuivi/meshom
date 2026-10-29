#!/bin/bash
# Download the public curvilinear ocean grids that scripts/grids/measure_grid.py measures.
#   scripts/grids/fetch_grids.sh [DATA_DIR]     (default ../curvilinear_grids next to meshom/)
# One subfolder per grid; existing files are kept (wget -nc). Sources and licences: DATA_DIR/SOURCES.md.
set -u
DATA=${1:-$(cd "$(git -C "$(dirname "$0")" rev-parse --path-format=absolute --git-common-dir)/../.." && pwd)/curvilinear_grids}
mkdir -p "$DATA"
get() {  # get SUBDIR URL [OUTNAME]
  mkdir -p "$DATA/$1"
  local out=${3:-$(basename "${2%%\?*}")}
  [ -s "$DATA/$1/$out" ] && return 0
  wget -q --tries=3 --timeout=120 -O "$DATA/$1/$out.part" "$2" && mv "$DATA/$1/$out.part" "$DATA/$1/$out" \
    || { echo "FAILED $1 $2"; rm -f "$DATA/$1/$out.part"; }
}
CEDA=https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP
SETTE=https://gws-access.jasmin.ac.uk/public/nemo/sette_inputs/r5.0.0
ZEN=https://zenodo.org/api/records
SCRIP=https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/share/scripgrids
GH=https://raw.githubusercontent.com
NAS=https://data.nas.nasa.gov/ecco/llc_270/grid

# NEMO
get nemo_orca2 $SETTE/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc
get nemo_eorca1 $ZEN/22010172/files/grid_mask_eORCA1-GO6.nc/content grid_mask_eORCA1-GO6.nc
get nemo_eorca025 $ZEN/15494369/files/grid_mask_eORCA025-GO6.nc/content grid_mask_eORCA025-GO6.nc
get nemo_eorca12 $ZEN/15495870/files/grid_mask_eORCA12-GO6.nc/content grid_mask_eORCA12-GO6.nc
get nemo_amm12 $SETTE/AMM12_v5.0.0/AMM_R12_sco_domcfg.nc
# Baltic MFC: only the 2D fields of the 4.3 GB domain_cfg.nc, read with HTTP range requests (needs numpy)
mkdir -p "$DATA/nemo_bal_mfc"
[ -s "$DATA/nemo_bal_mfc/domain_cfg_2d_extract.npz" ] || python -I "$(dirname "$0")/cdf_range.py" \
  $ZEN/14507734/files/domain_cfg.nc/content "$DATA/nemo_bal_mfc/domain_cfg_2d_extract.npz" \
  glamt gphit glamf gphif bottom_level jperio
get nemo_amm15 https://gws-access.jasmin.ac.uk/public/jmmp/AMM15/COORDINATES/amm15.coordinates.nc
get nemo_amm15 "https://gws-access.jasmin.ac.uk/public/jmmp/AMM15/AMM15_BATHY/ADD_NICO_BALTIC_BDY_RIM_SMOOTH_BDY_COPY_CUTAMM15_CORRECTED_EXPANDED_MERGE_GEBCO_DEEP_TO_200-100_EMODNET_TO_10-5_GEBCO_TO_COAST_amm15.bathydepth.co7.cs3x.cs20.nc" amm15.bathydepth.co7.cs3x.cs20.nc
# MITgcm
for f in 1 2 3 4 5 6; do get mitgcm_cs32 $GH/gaelforget/GRID_CS32/master/grid_cs32.face00$f.bin; done
for f in Depth.data Depth.meta; do get mitgcm_cs32 $GH/gaelforget/GRID_CS32/master/$f; done
for f in 1 2 3 4 5; do get mitgcm_llc90 $GH/gaelforget/GRID_LLC90/master/tile00$f.mitgrid; done
for f in Depth.data Depth.meta XG.data XG.meta YG.data YG.meta; do get mitgcm_llc90 $GH/gaelforget/GRID_LLC90/master/$f; done  # XG, YG: check of the compact reader
for f in XG YG XC YC Depth; do get mitgcm_llc270 $NAS/$f.data; get mitgcm_llc270 $NAS/$f.meta; done
# MOM6 (GFDL OM4, CMIP6)
get mom6_om4p25 $CEDA/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Ofx/sftof/gn/v20180701/sftof_Ofx_GFDL-CM4_piControl_r1i1p1f1_gn.nc
get mom6_om4p5 $CEDA/NOAA-GFDL/GFDL-ESM4/piControl/r1i1p1f1/Ofx/areacello/gn/v20180701/areacello_Ofx_GFDL-ESM4_piControl_r1i1p1f1_gn.nc
get mom6_om4p5 $CEDA/NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/Ofx/deptho/gn/v20190726/deptho_Ofx_GFDL-ESM4_historical_r1i1p1f1_gn.nc
# POP (CESM inputdata, SCRIP)
get pop_gx3v7 $SCRIP/gx3v7_120309.nc
get pop_gx1v6 $SCRIP/gx1v6_090205.nc
get pop_tx0.1v3 $SCRIP/tx0.1v3_170728.nc
# MPIOM (MPI-ESM1.2, CMIP6)
get mpiom_gr15 $CEDA/MPI-M/MPI-ESM1-2-LR/piControl/r1i1p1f1/Ofx/sftof/gn/v20190710/sftof_Ofx_MPI-ESM1-2-LR_piControl_r1i1p1f1_gn.nc
get mpiom_tp04 $CEDA/MPI-M/MPI-ESM1-2-HR/piControl/r1i1p1f1/Ofx/sftof/gn/v20190710/sftof_Ofx_MPI-ESM1-2-HR_piControl_r1i1p1f1_gn.nc
# HYCOM GLBb0.08
HY=https://data.hycom.org/datasets/GLBb0.08/expt_93.0/topo
# qlon, qlat = fields 3-4 of the 19 in regional.grid.a (4500 x 3298 padded to 14843904 float32 each)
[ -s "$DATA/hycom_glbb0.08/regional.grid.qlon_qlat.a" ] || { mkdir -p "$DATA/hycom_glbb0.08"; curl -s -r 118751232-237502463 \
  -o "$DATA/hycom_glbb0.08/regional.grid.qlon_qlat.a" $HY/regional.grid.a; }
get hycom_glbb0.08 $HY/regional.grid.b
get hycom_glbb0.08 $HY/regional_grid.nc
get hycom_glbb0.08 $HY/depth_GLBb0.08_09m11.a
get hycom_glbb0.08 $HY/depth_GLBb0.08_09m11.b
# ROMS
get roms_txla https://github.com/xoceanmodel/xroms/raw/main/xroms/data/ROMS_example_full_grid.nc
get roms_wc13 https://media.githubusercontent.com/media/myroms/roms_test/main/WC13/Data/wc13_grd.nc
get roms_usec3km https://media.githubusercontent.com/media/myroms/roms_test/main/USEC/Data/GRD/usec3km_roms_grd.nc4
# Delft3D-FLOW (Zenodo model setups)
get delft3d_rongjiang $ZEN/8420090/files/2D-3D_Model4Sharing.zip/content 2D-3D_Model4Sharing.zip
get delft3d_caernarvon "$ZEN/12639858/files/akhalifa75/1927_Delft3D_Model-1927_Delft3D_Model.zip/content" 1927_Delft3D_Model.zip
mkdir -p "$DATA/delft3d_rongjiang/unzipped" "$DATA/delft3d_caernarvon/unzipped"
unzip -q -n "$DATA/delft3d_rongjiang/2D-3D_Model4Sharing.zip" 'Grids/*' -d "$DATA/delft3d_rongjiang/unzipped"
unzip -q -n "$DATA/delft3d_caernarvon/1927_Delft3D_Model.zip" -d "$DATA/delft3d_caernarvon/unzipped"
unzip -q -n "$DATA"/delft3d_caernarvon/unzipped/*/Run-31-Shared.zip 'Run-31-Shared/BRE_C_cut5_div.*' \
  'Run-31-Shared/flow1.mdf' -d "$DATA/delft3d_caernarvon/unzipped"
du -sh "$DATA"
