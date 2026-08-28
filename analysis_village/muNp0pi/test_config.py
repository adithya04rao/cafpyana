from analysis_village.gump.makedf import *
#from makedf.make_mnp_df import make_recodf

DFS = [make_stubs, make_gump_nudf,
    # make_crthitdf, 
    make_hdrdf, make_triggerdf, make_potdf_bnb, make_opflashdf, make_mcprimdf, make_recodf, make_gump_nuwgtdf, make_slcdf]

NAMES = ["stub", "mcnu",
    # "crt", 
    "hdr", "trig", "bnb", "flash", "prim", "reco", "wgt", "slc"]