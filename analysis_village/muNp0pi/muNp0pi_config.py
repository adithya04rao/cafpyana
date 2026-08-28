from analysis_village.gump.makedf import *

DFS = [make_pandora_no_cuts_df, make_gump_nudf, make_stubs, 
    # make_crthitdf, 
    make_hdrdf, make_triggerdf, make_potdf_bnb, make_opflashdf, make_mcprimdf, make_recodf, make_gump_nuwgtdf]

NAMES = ["evt", "mcnu", "stub", 
    # "crt", 
    "hdr", "trig", "bnb", "flash", "prim", "reco", "wgt"] 