import os
import time
import pandas as pd
from analysis_village.gump.loaddf import *
from analysis_village.muNp0pi.muNp0pi_funcs import *
import gc

base = "/exp/sbnd/data/users/arao/SBND" ## The directory where your reconstructed df (in this case recodf) is stores

det = "SBND" ## SBND, ICARUS Run4 or ICARUS Run2

n_splits = 40                           ## Adjust these numbers based on the number of dfs you have and the number of sbruce files you want
files_per_split = 1000 // n_splits

start_time = time.time()
missing = []
pots_list = []
ntuple_offset = 0

print("========================================")
print("Starting processing")
print(f"Creating {n_splits} splits")
print(f"{files_per_split} input files per split")
print("========================================")

for split in range(n_splits):

    split_number = split + 1
    split_start = split * files_per_split
    split_end = (split + 1) * files_per_split

    print("\n========================================")
    print(f"STARTING SPLIT {split_number}/{n_splits}")
    print(f"Files: {split_start} -> {split_end - 1}")
    print("========================================")

    wgtdfs_split = []
    muNp0pidfs_split = []

    for i in range(split_start, split_end):

        filename = f"{base}/dfs/sbnd_arao_{i}.df"  ## Path to your HDF5 files

        if not os.path.exists(filename):
            print(f"Missing: {i}")
            missing.append(i)
            continue

        try:
            # ------------------------------------------
            # Load
            # ------------------------------------------

            wgtdf_i, match, pots_i = load(
                filename,
                detector=det,
                xsec_spline=True,
                drops=get_std_drops()
            )

            pots_list.append(pots_i)

            recodf_i = pd.read_hdf(
                filename,
                key="reco_0"  ## Key for the df you're using for the selection
            )

            # ------------------------------------------
            # Same ntuple offset for reco + wgtdf
            # ------------------------------------------

            ntuple = recodf_i.index.get_level_values("__ntuple")
            min_ntuple = ntuple.min()
            max_ntuple = ntuple.max()
            shift = ntuple_offset - min_ntuple

            recodf_i.index = pd.MultiIndex.from_arrays(
                [
                    ntuple + shift,
                    recodf_i.index.get_level_values("entry"),
                    recodf_i.index.get_level_values("rec.slc..index"),
                    recodf_i.index.get_level_values("rec.slc.reco.pfp..index")
                ],
                names=recodf_i.index.names
            )

            wgt_ntuple = wgtdf_i.index.get_level_values("__ntuple")

            wgtdf_i.index = pd.MultiIndex.from_arrays(
                [
                    wgt_ntuple + shift,
                    wgtdf_i.index.get_level_values("entry"),
                    wgtdf_i.index.get_level_values("rec.slc..index")
                ],
                names=wgtdf_i.index.names
            )

            ntuple_offset += max_ntuple - min_ntuple + 1

            print(
                f"File {i}: ntuple "
                f"{ntuple.min()}->{ntuple.max()} "
                f"-> {ntuple_offset - (max_ntuple - min_ntuple + 1)}"
                f"->{ntuple_offset - 1}"
            )

            # ------------------------------------------
            # Pipeline
            # ------------------------------------------

            muNp0pidf_i = muNp0pi_pipeline(
                recodf_i,
                30,                     ## Adjust cuts accordingly
                80,
                40,
                400,
                det
            )

            print(f"  wgtdf: {wgtdf_i.shape}")
            print(f"  muNp0pidf: {muNp0pidf_i.shape}")

            wgtdfs_split.append(wgtdf_i)
            muNp0pidfs_split.append(muNp0pidf_i)

            del recodf_i, wgtdf_i, muNp0pidf_i
            gc.collect()

        except Exception as e:
            print(f"\nERROR in file {i}:")
            print(filename)
            print(e)
            missing.append(i)
            continue

        elapsed = time.time() - start_time
        processed = i + 1
        rate = processed / elapsed
        remaining = (1000 - processed) / rate if rate > 0 else 0

        print(
            f"File {processed}/1000 | "
            f"Split {split_number}/{n_splits} | "
            f"Elapsed {elapsed/60:.1f} min | "
            f"ETA {remaining/60:.1f} min"
        )

    # ==================================================
    # Save wgtdf
    # ==================================================

    print("\n----------------------------------------")
    print(f"Combining wgtdf split {split_number}")
    print("----------------------------------------")

    if wgtdfs_split:
        wgtdf_split = pd.concat(wgtdfs_split, copy=False)
        print(f"wgtdf shape: {wgtdf_split.shape}")
        det.replace(' ', '_')
        wgtdf_output = f"{base}/wgtdf_{det}_{split_number:02d}.h5"

        wgtdf_split.reset_index().to_hdf(
            wgtdf_output,
            key="data",
            mode="w",
            format="fixed"
        )

        print(f"wgtdf saved: {wgtdf_output}")
        print(f"wgtdf size: {os.path.getsize(wgtdf_output)/1e9:.2f} GB")

        del wgtdf_split
        gc.collect()

    # ==================================================
    # Save muNp0pidf
    # ==================================================

    print("\n----------------------------------------")
    print(f"Combining muNp0pidf split {split_number}")
    print("----------------------------------------")

    if muNp0pidfs_split:
        muNp0pidf_split = pd.concat(
            muNp0pidfs_split,
            copy=False
        )

        print(f"muNp0pidf shape: {muNp0pidf_split.shape}")

        mu_output = f"{base}/muNp0pidf_{det}_{split_number:02d}.h5"

        muNp0pidf_split.reset_index().to_hdf(
            mu_output,
            key="data",
            mode="w",
            format="fixed"
        )

        print(f"muNp0pidf saved: {mu_output}")
        print(f"muNp0pidf size: {os.path.getsize(mu_output)/1e6:.1f} MB")

        del muNp0pidf_split
        gc.collect()

    print("\n========================================")
    print(f"FINISHED SPLIT {split_number}/{n_splits}")
    print("========================================")


# ==================================================
# Final
# ==================================================

pots = sum(pots_list)
elapsed = time.time() - start_time

print("\n========================================")
print("COMPLETE")
print("========================================")
print(f"Successful files: {20 - len(missing):,}")
print(f"Missing/bad files: {len(missing):,}")
print(f"POT: {pots:,}")
print(f"Total time: {elapsed/60:.1f} minutes")

print("\nCreated files:")

for split in range(1, n_splits + 1):

    wgtdf_output = f"{base}/wgtdf_{det}_{split:02d}.h5"
    mu_output = f"{base}/muNp0pidf_{det}_{split:02d}.h5"

    if os.path.exists(wgtdf_output):
        print(
            f"{os.path.basename(wgtdf_output)}: "
            f"{os.path.getsize(wgtdf_output)/1e9:.2f} GB"
        )

    if os.path.exists(mu_output):
        print(
            f"{os.path.basename(mu_output)}: "
            f"{os.path.getsize(mu_output)/1e6:.1f} MB"
        )

if missing:
    print("\nMissing/bad files:")
    print(missing)


import gc

print("========================================")
print("Creating muNp0pi_wgtdf")
print("========================================")


index_cols = [
    "__ntuple",
    "entry",
    "rec.slc..index"
]

right_cols = [
    "slc_reco_E",
    "slc_truth_q0_lab",
    "slc_truth_baseline",
    "slc_truth_E",
    "slc_truth_modq_lab",
    "slc_truth_np",
    "slc_truth_npi",
    "slc_reco_np",
    "pfp_trk_truth_p_interaction_id"
]

for split in range(1, n_splits + 1):

    print("\n========================================")
    print(f"STARTING SPLIT {split}/{n_splits}")
    print("========================================")

    # --------------------------------------------------
    # File names
    # --------------------------------------------------

    mu_file = os.path.expanduser(
        f"{base}/muNp0pidf_{det}_{split:02d}.h5"
    )

    wgt_file = os.path.expanduser(
        f"{base}/wgtdf_{det}_{split:02d}.h5"
    )

    output_file = os.path.expanduser(
        f"{base}/pkls/muNp0pi_wgtdf_{det}_{split:02d}.pkl"
    )

    # --------------------------------------------------
    # Check inputs
    # --------------------------------------------------

    if not os.path.exists(mu_file):
        print(f"Missing: {mu_file}")
        continue

    if not os.path.exists(wgt_file):
        print(f"Missing: {wgt_file}")
        continue

    # --------------------------------------------------
    # Load muNp0pidf
    # --------------------------------------------------

    print("Loading muNp0pidf...")

    muNp0pidf = pd.read_hdf(
        mu_file,
        key="data"
    )

    print(
        f"muNp0pidf rows: "
        f"{len(muNp0pidf):,}"
    )

    # Reset index for explicit merge
    muNp0pidf = muNp0pidf.reset_index()

    # Keep only required columns
    muNp0pidf_right = muNp0pidf[
        index_cols + right_cols
    ]

    del muNp0pidf

    gc.collect()

    # --------------------------------------------------
    # Load wgtdf
    # --------------------------------------------------

    print("Loading wgtdf...")

    wgtdf = pd.read_hdf(
        wgt_file,
        key="data"
    )

    print(
        f"wgtdf rows: "
        f"{len(wgtdf):,}"
    )

    # Reset index for explicit merge
    wgtdf = wgtdf.reset_index()

    # --------------------------------------------------
    # Find matches
    # --------------------------------------------------

    print("Finding matching rows...")

    result = muNp0pidf_right.merge(
        wgtdf,
        on=index_cols,
        how="inner",
        sort=False,
        suffixes=("", "_wgt")
    )

    print(
        f"Matching rows: "
        f"{len(result):,}"
    )

    print(
        f"Result shape: "
        f"{result.shape}"
    )

    # --------------------------------------------------
    # Verify matches
    # --------------------------------------------------

    mu_keys = set(
        map(
            tuple,
            muNp0pidf_right[index_cols].drop_duplicates().values
        )
    )

    wgt_keys = set(
        map(
            tuple,
            wgtdf[index_cols].drop_duplicates().values
        )
    )

    result_keys = set(
        map(
            tuple,
            result[index_cols].drop_duplicates().values
        )
    )

    print(
        f"Events in muNp0pidf: "
        f"{len(mu_keys):,}"
    )

    print(
        f"Events in wgtdf: "
        f"{len(wgt_keys):,}"
    )

    print(
        f"Events in both: "
        f"{len(mu_keys & wgt_keys):,}"
    )

    print(
        f"Events in result: "
        f"{len(result_keys):,}"
    )

    print(
        f"Result events not in wgtdf: "
        f"{len(result_keys - wgt_keys)}"
    )

    print(
        f"Result events not in muNp0pidf: "
        f"{len(result_keys - mu_keys)}"
    )

    # --------------------------------------------------
    # Save as pickle
    # --------------------------------------------------

    print("Saving...")

    result.to_pickle(
        output_file
    )

    # --------------------------------------------------
    # Verify
    # --------------------------------------------------

    if os.path.exists(output_file):

        size_gb = (
            os.path.getsize(output_file)
            / 1e9
        )

        print(
            "Saved successfully:"
        )

        print(output_file)

        print(
            f"Size: {size_gb:.2f} GB"
        )

    else:

        print(
            "ERROR: Output file was not created."
        )

        continue

    # --------------------------------------------------
    # Delete input split files
    # --------------------------------------------------

    print("Deleting input HDF5 files...")

    os.remove(mu_file)

    print(
        f"Deleted {mu_file}"
    )

    os.remove(wgt_file)

    print(
        f"Deleted {wgt_file}"
    )

    # --------------------------------------------------
    # Free memory
    # --------------------------------------------------

    del muNp0pidf_right
    del wgtdf
    del result
    del mu_keys
    del wgt_keys
    del result_keys

    gc.collect()

    print(
        f"Finished split "
        f"{split}/{n_splits}"
    )


print("\n========================================")
print("ALL MATCHING COMPLETE")
print("========================================")

print("\nFinal files:")

for split in range(1, n_splits + 1):

    output_file = os.path.expanduser(
        f"{base}/pkls/muNp0pi_wgtdf_{det}_{split:02d}.pkl"
    )

    if os.path.exists(output_file):

        print(
            f"{os.path.basename(output_file)}: "
            f"{os.path.getsize(output_file) / 1e9:.2f} GB"
        )

import os
import pandas as pd

from analysis_village.gump.sbruce import (
    export_dataframe_to_uproot,
    run_makesbruce_macro
)

import gc

macro_dir = (
    "/exp/sbnd/data/users/arao/"
    "cafpyana/analysis_village/PROfit/"
)

for i in range(1, 41):

    print("\n========================================")
    print(f"PROCESSING SPLIT {i}/40")
    print("========================================")

    # --------------------------------------------
    # File names
    # --------------------------------------------

    pkl_file = (
        f"{base}/pkls/muNp0pi_wgtdf_{det}_{i:02d}.pkl"
    )

    root_file = (
        f"{base}/root/muNp0pi_wgtdf_{det}_{i:02d}.root"
    )

    sbruce_file = (
        f"{base}/sbruce/muNp0pi_wgtdf_{det}_sbruce_{i:02d}.root"
    )

    # --------------------------------------------
    # Check input
    # --------------------------------------------

    if not os.path.exists(pkl_file):

        print(f"Missing: {pkl_file}")
        continue

    print(f"Input: {pkl_file}")

    # --------------------------------------------
    # Load one pickle
    # --------------------------------------------

    print("Loading pickle...")

    muNp0pi_wgtdf = pd.read_pickle(
        pkl_file
    )

    print(
        f"Loaded shape: "
        f"{muNp0pi_wgtdf.shape}"
    )

    # --------------------------------------------
    # Export to ROOT
    # --------------------------------------------

    print("Exporting to ROOT...")

    export_dataframe_to_uproot(
        muNp0pi_wgtdf,
        root_file
    )

    print(f"Created: {root_file}")

    # --------------------------------------------
    # Run sbruce
    # --------------------------------------------

    print("Running makesbruce macro...")

    run_makesbruce_macro(
        root_file,
        sbruce_file,
        macro_dir=macro_dir
    )

    print(f"Created: {sbruce_file}")

    # --------------------------------------------
    # Free memory
    # --------------------------------------------

    del muNp0pi_wgtdf
    gc.collect()

    print(f"FINISHED SPLIT {i}/40")


print("\n========================================")
print("ALL 40 SPLITS COMPLETE")
print("========================================")