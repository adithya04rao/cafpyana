from analysis_village.gump.makedf import *
from analysis_village.muNp0pi.makedf import make_muNp0pidf

DFS = [make_pandora_no_cuts_df, make_gump_nudf, make_stubs, 
    # make_crthitdf, 
    make_hdrdf, make_triggerdf, make_potdf_bnb, make_opflashdf, make_mcprimdf, make_recodf, make_gump_nuwgtdf, make_muNp0pidf]

NAMES = ["evt", "mcnu", "stub", 
    # "crt", 
    "hdr", "trig", "bnb", "flash", "prim", "reco", "wgt", "muNp"]
