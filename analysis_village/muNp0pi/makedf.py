import functools
from functools import reduce
from pyanalib.pandas_helpers import *
from makedf.branches import *
from makedf.util import *
from makedf.calo import *
from makedf import numisyst, g4syst, geniesyst, bnbsyst, getenv
from makedf import chi2pid
import pyanalib.pandas_helpers as ph
from makedf import branches
import makedf.util as util
from analysis_village.muNp0pi.muNp0pi_cuts import all_fv_cuts
from analysis_village.muNp0pi.muNp0pi_funcs import muNp0pi_pipeline

pd.set_option('future.no_silent_downcasting', True)

PDG = {
    "muon": [13, "muon", 0.105,],
    "proton": [2212, "proton", 0.938272,],
    "neutron": [2112, "neutron", 0.9395654,],
    "pizero": [111, "pizero", 0.1349768],
    "pipm": [211, "piplus", 0.13957039],
    "argon": [1000180400, "argon", (18*0.938272 + 22*0.9395654)],
    "gamma": [22, "gamma", 0 ],
    "lambda": [3122, "lambda", 1.115683],
    "kaon_p": [321, "kaon_p",  0.493677],
    "sigma_p": [3222, "sigma_p", 1.18936],
    "kaon_0": [311, "kaon_0", 0.497648],
    "sigma_0": [3212, "sigma_0", 1.19246],
    "lambda_p_c": [4122, "lambda_p_c", 2.28646],
    "sigma_pp_c": [4222, "sigma_pp_c", 2.45397],
    "electron": [11, "electron", 0.000510998950],
    "sigma_p_c": [4212, "sigma_p_c", 2.4529],
}

## == For additional column in mcdf with primary particle multiplicities
## ==== "<column name>": ["<particle name>", <KE cut in GeV>]
## ==== <particle name> is used to collect PID and mass from the "PDG" dictionary
TRUE_KE_THRESHOLDS = {"nmu_27MeV": ["muon", 0.027],
                      "np_20MeV": ["proton", 0.02],
                      "np_50MeV": ["proton", 0.05],
                      "npi_30MeV": ["pipm", 0.03],
                      "nn_0MeV": ["neutron", 0.0]
                      }

def make_envdf(f):
    env = getenv.get_env(f)
    return env

def make_histpotdf(f):
    if f is None or "TotalPOT" not in f:
        histpot = pd.DataFrame({"TotalPOT": pd.Series(dtype="float64")})
        histpot.index.name = "entry"
        return histpot

    pot = f['TotalPOT'].values()
    histpot = pd.DataFrame(data={'TotalPOT':pot})
    histpot.index.name = 'entry'
    return histpot

def make_histgenevtdf(f):
    if f is None or "TotalGenEvents" not in f:
        histgenevt = pd.DataFrame({"TotalGenEvents": pd.Series(dtype="float64")})
        histgenevt.index.name = "entry"
        return histgenevt

    genevt = f['TotalGenEvents'].values()
    histgenevt = pd.DataFrame(data={'TotalGenEvents':genevt})
    histgenevt.index.name = 'entry'
    return histgenevt

def make_hdrdf(f):
    hdr = loadbranches(f["recTree"], hdrbranches).rec.hdr
    return hdr

def make_mchdrdf(f):
    hdr = loadbranches(f["recTree"], mchdrbranches).rec.hdr
    return hdr

def make_potdf_bnb(f):
    pot = loadbranches(f["recTree"], bnbpotbranches).rec.hdr.bnbinfo
    return pot

def make_potdf_numi(f):
    pot = loadbranches(f["recTree"], numipotbranches).rec.hdr.numiinfo
    return pot

def make_framedf(f):
    frame = loadbranches(f["recTree"],sbndframebranches).rec.sbnd_frames
    return frame

def make_timingdf(f):
    timing = loadbranches(f["recTree"],sbndtimingbranches).rec.sbnd_timings
    return timing

def make_triggerdf(f):
    return  loadbranches(f["recTree"], trigger_info_branches).rec.hdr.triggerinfo

def make_mcnuwgtdf(f):
    return make_mcnudf(f, include_weights=True, multisim_nuniv=100)

def make_mcnuwgtdf_slim(f):
    return make_mcnudf(f, include_weights=True, multisim_nuniv=100, slim=True)

# TODO: zip the nuniv configs
def make_mcnudf(f, include_weights=False, multisim_nuniv=100, genie_multisim_nuniv=100, wgt_types=["bnb","genie","g4"], slim=False, genie_systematics=None):
    # ----- sbnd or icarus? -----
    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    if (1 == det.unique()):
        det = "SBND"
    else:
        det = "ICARUS"

    mcdf = make_mcdf(f)
    mcdf["ind"] = mcdf.index.get_level_values(1)
    if include_weights:
        if len(wgt_types) == 0:
            print("include_weights is set to True, pass at least one type of wgt to save")
        else:
            df_list = []
            if "bnb" in wgt_types:
                bnbwgtdf = bnbsyst.bnbsyst(f, mcdf.ind, multisim_nuniv=multisim_nuniv, slim=slim)
                df_list.append(bnbwgtdf)
            if "genie" in wgt_types:
                geniewgtdf = geniesyst.geniesyst(f, mcdf.ind, multisim_nuniv=genie_multisim_nuniv, slim=slim, systematics=genie_systematics)
                df_list.append(geniewgtdf)
            if "g4" in wgt_types:
                g4wgtdf = g4syst.g4syst(f, mcdf.ind)
                df_list.append(g4wgtdf)

            wgtdf = pd.concat(df_list, axis=1)
            mcdf = multicol_concat(mcdf, wgtdf)

    return mcdf  

def make_mevprtlwgtdf(f):
    return make_mevprtldf(f, include_weights=True, multisim_nuniv=100)

def make_mevprtlwgtdf_slim(f):
    return make_mevprtldf(f, include_weights=True, multisim_nuniv=100, slim=True)

def make_mevprtldf(f, branches = mevprtltruthbranches, include_weights=False, multisim_nuniv=100, genie_multisim_nuniv=100, wgt_types=["bnb","g4"], slim=False):
    # ----- sbnd or icarus? -----
    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    if (1 == det.unique()):
        det = "SBND"
    else:
        det = "ICARUS"

    mcdf = loadbranches(f["recTree"], branches).rec.mc.prtl
    mcdf["ind"] = mcdf.index.get_level_values(1)
    if include_weights:
        if len(wgt_types) == 0:
            print("include_weights is set to True, pass at least one type of wgt to save")
        else:
            df_list = []
            if "bnb" in wgt_types:
                bnbwgtdf = bnbsyst.bnbsyst(f, mcdf.ind, multisim_nuniv=multisim_nuniv, slim=slim)
                df_list.append(bnbwgtdf)
            if "g4" in wgt_types:
                g4wgtdf = g4syst.g4syst(f, mcdf.ind)
                df_list.append(g4wgtdf)

            wgtdf = pd.concat(df_list, axis=1)
            mcdf = multicol_concat(mcdf, wgtdf)
    return mcdf

def make_geniedf(f):
    if "GenieEvtRecTree" not in f:
        return pd.DataFrame([])

    # shape = n of particles in genie 
    genie_particle_branches = [
        "GenieEvtRec.StdHepPdg",
        "GenieEvtRec.StdHepStatus",
        "GenieEvtRec.StdHepFm",
    ]
    # shape = 1 (per event)
    genie_event_branches = [
        "GENIEEntry",
        "SourceFileHash",
        "GenieEvtRec.EvtNum",
        "GenieEvtRec.StdHepN",
    ]
    # Branches all have different shapes, need to manipulate before merging 
    p_df = loadbranches(f["GenieEvtRecTree"],genie_particle_branches)
    # shape = 4-vector per particle 
    m_df = loadbranches(f["GenieEvtRecTree"],["GenieEvtRec.StdHepP4",])
    m_df = m_df.unstack().rename(columns={0: 'px', 1 :'py', 2 :'pz', 3:'E'}, level=2)
    p_df = multicol_merge(p_df,m_df,left_index=True,right_index=True)

    e_df = loadbranches(f["GenieEvtRecTree"],genie_event_branches)
    p_df = multicol_merge(e_df,p_df,left_index=True,right_index=True) 
    
    # shape = 4-vector per event
    v_df = loadbranches(f["GenieEvtRecTree"], ["GenieEvtRec.EvtVtx"])
    v_df = v_df.unstack().rename(columns={0: 'x', 1 :'y', 2 :'z', 3:'E'}, level=2)

    df = multicol_merge(v_df,p_df,left_index=True,right_index=True)
    df = df.reset_index().set_index('entry')
    df = df.rename(columns={'subentry': 'pindex'},level=0)
    return df

def make_mchdf(f, include_weights=False):
    mcdf = loadbranches(f["recTree"], mchbranches).rec.mc.prtl
    if include_weights:
        wgtdf = numisyst.numisyst(14, mcdf.E) # TODO: what PDG?
        mcdf = pd.concat([mcdf, wgtdf], axis=1)
    return mcdf

def make_crtspdf(f):
    crtspdf = loadbranches(f["recTree"], crtspbranches).rec
    return crtspdf

def make_crtvetodf(f):
    crtvetodf = loadbranches(f["recTree"], crtvetobranches).rec
    return crtvetodf

def make_crthitdf(f):
    crthitdf = loadbranches(f["recTree"], crthitbranches).rec.crt_hits
    return crthitdf

def make_opflashdf(f):
    opflashdf = loadbranches(f["recTree"], opflashbranches).rec.opflashes
    return opflashdf

def make_trkdf(f, scoreCut=False, requiret0=False, requireCosmic=False, mcs=False, det="SBND", updatecalo=None):
    trkdf = loadbranches(f["recTree"], trkbranches)
    if scoreCut:
        trkdf = trkdf.rec.slc.reco[trkdf.rec.slc.reco.pfp.trackScore > 0.5]
    else:
        trkdf = trkdf.rec.slc.reco

    if requiret0:
        trkdf = trkdf[~np.isnan(trkdf.pfp.t0)]

    if requireCosmic:
        trkdf = trkdf[trkdf.pfp.parent == -1]

    if mcs:
        mcsdf = loadbranches(f["recTree"], [trkmcsbranches[0]]).rec.slc.reco.pfp.trk.mcsP
        mcsdf_angle = loadbranches(f["recTree"], [trkmcsbranches[1]]).rec.slc.reco.pfp.trk.mcsP
        mcsdf_angle.index.set_names(mcsdf.index.names, inplace=True)

        mcsdf = mcsdf.merge(mcsdf_angle, how="left", left_index=True, right_index=True)
        mcsgroup = list(range(mcsdf.index.nlevels-1))
        cumlen = mcsdf.seg_length.groupby(level=mcsgroup).cumsum()*14 # convert rad length to cm
        maxlen = (cumlen*(mcsdf.seg_scatter_angles >= 0)).groupby(level=mcsgroup).max()
        trkdf[("pfp", "trk", "mcsP", "len", "", "")] = maxlen
    trkdf[("pfp", "tindex", "", "", "", "")] = trkdf.index.get_level_values(2)


    if updatecalo is not None:
        hdrdf = make_mchdrdf(f)
        ismc = hdrdf.ismc.iloc[0]

        for plane in range(0, 3):
            trkhitdf = make_trkhitdf(f, plane)
            trkhitdf = trkhitdf[InFV(df=trkhitdf, det=det)]

            dedx_redo = chi2pid.dedx(trkhitdf, gain=det, calibrate=det, plane=plane, isMC=ismc, new_calo_params=chi2pid.CALO_VARIATIONS[updatecalo])

            trkhitdf["dedx_redo"] = dedx_redo
            # TODO: check if score is reproduced
            # dedx_bias = (dedx_redo - trkhitdf.dedx) / trkhitdf.dedx
            # trkhitdf["dedx_bias"] = dedx_bias
            # print("bias", list(dedx_bias.head()))
            for par in ['muon', 'proton']:
                this_chi2_new, this_chi2_ndof = chi2pid.chi2par(trkhitdf, dedxname="dedx_redo", par=par)
                this_chi2_col = ('pfp', 'trk', 'chi2pid', 'I' + str(plane), 'chi2_' + par + '_new', '')
                this_ndof_col = ('pfp', 'trk', 'chi2pid', 'I' + str(plane), 'ndof_' + par + '_new', '')
                trkdf[this_chi2_col] = this_chi2_new.fillna(-999)
                trkdf[this_ndof_col] = this_chi2_ndof.fillna(-999)

    return trkdf

def make_pfpdf(f, update_shw=True):
    pfpdf = loadbranches(f["recTree"], trkbranches + shwbranches)
    pfpdf = pfpdf.rec.slc.reco
    
    if update_shw:
        ## necessary since "bestplane" stored in the cafs currently is from the dEdx alg
        ## bestplane for dEdx alg is not necessarily the same as bestplane for shower energy

        # set shower energy as the one with the plane that has the most number of hits (maxplane)
        pfpdf['pfp','shw','maxplane','','',''] = pfpdf.loc(axis=1)['pfp','shw','plane',:,"nHits"].idxmax(axis=1).apply(lambda x: x[3])
        pfpdf['pfp','shw','maxplane_energy','','',''] = np.nan
        conditions = [pfpdf['pfp','shw','maxplane','','','']=="I2",pfpdf['pfp','shw','maxplane','','','']=="I1",pfpdf['pfp','shw','maxplane','','','']=="I0"]
        choices = [pfpdf['pfp','shw','plane','I2','energy',''],pfpdf['pfp','shw','plane','I1','energy',''],pfpdf['pfp','shw','plane','I0','energy','']]
        pfpdf['pfp','shw','maxplane_energy','','',''] = np.select(conditions,choices,default=np.nan)
    pfpdf[("pfp", "tindex", "", "", "", "")] = pfpdf.index.get_level_values(2)
    return pfpdf

def make_trkhitdf_plane0(f):
    return make_trkhitdf(f, 0)

def make_trkhitdf_plane1(f):
    return make_trkhitdf(f, 1)

def make_trkhitdf_plane2(f):
    return make_trkhitdf(f, 2)

def make_trkhitdf(f, plane=2):
    # ----- sbnd or icarus? -----
    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    if (1 == det.unique()):
        det = "SBND"
    else:
        det = "ICARUS"

    branches = [trkhitbranches_P0, trkhitbranches_P1, trkhitbranches][plane] if det == "SBND" else [trkhitbranches_P0_icarus, trkhitbranches_P1_icarus, trkhitbranches_icarus][plane]
    df = loadbranches(f["recTree"], branches).rec.slc.reco.pfp.trk.calo
    df = df["I" + str(plane)].points

    # get the cryostat
    df = df.merge(loadbranches(f["recTree"], ["rec.slc.reco.pfp.trk.producer"]).rec.slc.reco.pfp.trk.producer.rename("cryo"),  how="left", left_index=True, right_index=True)

    # save the plane
    df["plane"] = plane

    # Add in the run, useful in calibrations
    df = df.merge(loadbranches(f["recTree"], ["rec.hdr.run"]).rec.hdr, how="left", left_index=True, right_index=True)

    # Add in the track phi angle
    #
    # TODO: (when ready) -- get this from the hitdf for ICARUS, SBND is ready
    if det == "ICARUS":
        with np.errstate(invalid='ignore'):
            df = df.merge(np.arccos(np.abs(loadbranches(f["recTree"], ["rec.slc.reco.pfp.trk.dir.x"]).rec.slc.reco.pfp.trk.dir.x)).rename("phi"), how="left", left_index=True, right_index=True)

    # Add in the efield
    #
    # TODO: (when ready) -- get this from the hitdf for ICARUS, SBND is ready
    if det == "ICARUS":
        df["efield"] = Efield_icarus

    # and the density
    df["rho"] = LAr_density_gmL_icarus if (det == "ICARUS") else LAr_density_gmL_sbnd

    # Firsthit and Lasthit info
    ihit = df.index.get_level_values(-1)
    df["firsthit"] = ihit == 0

    lasthit = df.groupby(level=list(range(df.index.nlevels-1))).tail(1).copy()
    lasthit["lasthit"] = True
    df["lasthit"] = lasthit.lasthit
    df.lasthit = df.lasthit.fillna(False).infer_objects()

    return df

def make_trktruehitdf_plane0(f):
    return make_trktruehitdf(f, 0)

def make_trktruehitdf_plane1(f):
    return make_trktruehitdf(f, 1)

def make_trktruehitdf_plane2(f):
    return make_trktruehitdf(f, 2)

def make_trktruehitdf(f, plane=2):
    branches = [trktruehitbranches_P0, trktruehitbranches_P1, trktruehitbranches][plane]
    df = loadbranches(f["recTree"], branches).rec.slc.reco.pfp.trk.calo
    df = df["I" + str(plane)].points.truth

    return df

def make_slcdf(f):
    slcdf = loadbranches(f["recTree"], slcbranches)
    slcdf = slcdf.rec
    slc_mcdf = make_mcdf(f, slc_mcbranches, slc_mcprimbranches)
    slc_mcdf.columns = pd.MultiIndex.from_tuples([tuple(["slc", "truth"] + list(c)) for c in slc_mcdf.columns])
    slcdf = multicol_merge(slcdf, slc_mcdf, left_index=True, right_index=True, how="left", validate="one_to_one")

    return slcdf

def make_mcdf(f, branches=mcbranches, primbranches=mcprimbranches):
    # load the df
    mcdf = loadbranches(f["recTree"], branches)
    while mcdf.columns.nlevels > 2:
        mcdf.columns = mcdf.columns.droplevel(0)

    # Add in primary particle info
    mcprimdf = loadbranches(f["recTree"], primbranches)
    while mcprimdf.columns.nlevels > 2:
        mcprimdf.columns = mcprimdf.columns.droplevel(0)

    mcprimdf.index = mcprimdf.index.rename(mcdf.index.names[:2] + mcprimdf.index.names[2:])

    max_proton_KE = mcprimdf[np.abs(mcprimdf.pdg)==PDG["proton"][0]].genE.groupby(level=[0,1]).max() - PDG["proton"][2]
    mcdf = multicol_add(mcdf, max_proton_KE.rename("max_proton_ke"), default=0.)

    mcdf.max_proton_ke = mcdf.max_proton_ke.fillna(0.)

    # particle counts
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==2112).groupby(level=[0,1]).sum().rename("nn"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==2212).groupby(level=[0,1]).sum().rename("np"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==13).groupby(level=[0,1]).sum().rename("nmu"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==211).groupby(level=[0,1]).sum().rename("npi"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==111).groupby(level=[0,1]).sum().rename("npi0"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==22).groupby(level=[0,1]).sum().rename("ng"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==321).groupby(level=[0,1]).sum().rename("nk"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==310).groupby(level=[0,1]).sum().rename("nk0"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==3112).groupby(level=[0,1]).sum().rename("nsm"))
    mcdf = multicol_add(mcdf, (np.abs(mcprimdf.pdg)==3222).groupby(level=[0,1]).sum().rename("nsp"))

    # particle counts w/ threshold
    for identifier, (particle, threshold) in TRUE_KE_THRESHOLDS.items():
        this_KE = mcprimdf[np.abs(mcprimdf.pdg)==PDG[particle][0]].genE - PDG[particle][2]
        mcdf = multicol_add(mcdf, ((np.abs(mcprimdf.pdg)==PDG[particle][0]) & (this_KE > threshold)).groupby(level=[0,1]).sum().rename(identifier))
 
    # muon info
    mudf = mcprimdf[np.abs(mcprimdf.pdg)==13].sort_values(mcprimdf.index.names[:2] + [("genE", "")]).groupby(level=[0,1]).last()
    mudf.columns = pd.MultiIndex.from_tuples([tuple(["mu"] + list(c)) for c in mudf.columns])

    cpidf = mcprimdf[np.abs(mcprimdf.pdg)==211].sort_values(mcprimdf.index.names[:2] + [("genE", "")]).groupby(level=[0,1]).last()
    cpidf.columns = pd.MultiIndex.from_tuples([tuple(["cpi"] + list(c)) for c in cpidf.columns])

    pdf = mcprimdf[mcprimdf.pdg==2212].sort_values(mcprimdf.index.names[:2] + [("genE", "")]).groupby(level=[0,1]).last()
    pdf.columns = pd.MultiIndex.from_tuples([tuple(["p"] + list(c)) for c in pdf.columns])

    # sub-leading proton
    p2df = mcprimdf[mcprimdf.pdg==2212].sort_values(mcprimdf.index.names[:2] + [("genE", "")]).groupby(level=[0,1]).tail(2).groupby(level=[0,1]).first()
    p2df.columns = pd.MultiIndex.from_tuples([tuple(["p2"] + list(c)) for c in p2df.columns])
    p2df = p2df[p2df.p2.length != pdf.p.length] # remove single-proton slices

    # electron info
    edf = mcprimdf[np.abs(mcprimdf.pdg)==11].sort_values(mcprimdf.index.names[:2] + [("genE", "")]).groupby(level=[0,1]).last()
    edf.columns = pd.MultiIndex.from_tuples([tuple(["e"] + list(c)) for c in edf.columns])

    mcdf = multicol_merge(mcdf, mudf, left_index=True, right_index=True, how="left", validate="one_to_one")
    mcdf = multicol_merge(mcdf, cpidf, left_index=True, right_index=True, how="left", validate="one_to_one")
    mcdf = multicol_merge(mcdf, pdf, left_index=True, right_index=True, how="left", validate="one_to_one")
    mcdf = multicol_merge(mcdf, p2df, left_index=True, right_index=True, how="left", validate="one_to_one")
    mcdf = multicol_merge(mcdf, edf, left_index=True, right_index=True, how="left", validate="one_to_one")

    # primary track variables
    mcdf.loc[:, ('mu','totp','')] = np.sqrt(mcdf.mu.genp.x**2 + mcdf.mu.genp.y**2 + mcdf.mu.genp.z**2)
    mcdf.loc[:, ('e','totp','')] = np.sqrt(mcdf.e.genp.x**2 + mcdf.e.genp.y**2 + mcdf.e.genp.z**2)
    mcdf.loc[:, ('cpi','totp','')] = np.sqrt(mcdf.cpi.genp.x**2 + mcdf.cpi.genp.y**2 + mcdf.cpi.genp.z**2)
    mcdf.loc[:, ('p','totp','')] = np.sqrt(mcdf.p.genp.x**2 + mcdf.p.genp.y**2 + mcdf.p.genp.z**2)
    mcdf.loc[:, ('p2','totp','')] = np.sqrt(mcdf.p2.genp.x**2 + mcdf.p2.genp.y**2 + mcdf.p2.genp.z**2)

    # opening angles
    mcdf.loc[:, ('mu','dir','x')] = mcdf.mu.genp.x/mcdf.mu.totp
    mcdf.loc[:, ('mu','dir','y')] = mcdf.mu.genp.y/mcdf.mu.totp
    mcdf.loc[:, ('mu','dir','z')] = mcdf.mu.genp.z/mcdf.mu.totp
    mcdf.loc[:, ('e','dir','x')] = mcdf.e.genp.x/mcdf.e.totp
    mcdf.loc[:, ('e','dir','y')] = mcdf.e.genp.y/mcdf.e.totp
    mcdf.loc[:, ('e','dir','z')] = mcdf.e.genp.z/mcdf.e.totp
    mcdf.loc[:, ('cpi','dir','x')] = mcdf.cpi.genp.x/mcdf.cpi.totp
    mcdf.loc[:, ('cpi','dir','y')] = mcdf.cpi.genp.y/mcdf.cpi.totp
    mcdf.loc[:, ('cpi','dir','z')] = mcdf.cpi.genp.z/mcdf.cpi.totp
    mcdf.loc[:, ('p','dir','x')] = mcdf.p.genp.x/mcdf.p.totp
    mcdf.loc[:, ('p','dir','y')] = mcdf.p.genp.y/mcdf.p.totp
    mcdf.loc[:, ('p','dir','z')] = mcdf.p.genp.z/mcdf.p.totp
    mcdf.loc[:, ('p2','dir','x')] = mcdf.p2.genp.x/mcdf.p2.totp
    mcdf.loc[:, ('p2','dir','y')] = mcdf.p2.genp.y/mcdf.p2.totp
    mcdf.loc[:, ('p2','dir','z')] = mcdf.p2.genp.z/mcdf.p2.totp

    # endpoints
    mcdf.loc[:, ('mu','end','x')] = mcdf.mu.end.x
    mcdf.loc[:, ('mu','end','y')] = mcdf.mu.end.y
    mcdf.loc[:, ('mu','end','z')] = mcdf.mu.end.z
    mcdf.loc[:, ('p','end','x')] = mcdf.p.end.x
    mcdf.loc[:, ('p','end','y')] = mcdf.p.end.y
    mcdf.loc[:, ('p','end','z')] = mcdf.p.end.z

    return mcdf

def make_mcprimdf(f):
    mcprimdf = loadbranches(f["recTree"], mcprimbranches)
    return mcprimdf

def make_mcprimvisEdf(f):
    mcprimvisEdf = loadbranches(f["recTree"], mcprimvisEbranches)
    return mcprimvisEdf

def make_mcprimdaughtersdf(f):
    mcprimdaughtersdf = loadbranches(f["recTree"], mcprimdaughtersbranches)
    return mcprimdaughtersdf

def make_all_pandora_df(f):
    pfpdf = make_pfpdf(f)
    slcdf = make_slcdf(f)

    slcdf = multicol_merge(slcdf, pfpdf, left_index=True, right_index=True, how="right", validate="one_to_many")

    # distance from vertex to track/shower start
    slcdf = multicol_add(slcdf, dmagdf(slcdf.slc.vertex, slcdf.pfp.trk.start).rename(("pfp", "trk", "dist_to_vertex")))
    slcdf = multicol_add(slcdf, dmagdf(slcdf.slc.vertex, slcdf.pfp.shw.start).rename(("pfp", "shw", "dist_to_vertex")))

    return pfpdf

def make_pandora_df_calo_update(f, **trkArgs):
    pandoradf = make_pandora_df(f, trkScoreCut=False, trkDistCut=50., cutClearCosmic=True, requireFiducial=False, updatecalo=True, **trkArgs)
    return pandoradf

def make_pandora_df(f, trkScoreCut=False, trkDistCut=50., cutClearCosmic=False, requireFiducial=False, updatecalo=False, **trkArgs):
    # load
    trkdf = make_trkdf(f, trkScoreCut, **trkArgs)
    if updatecalo:
        # check detector
        det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
        if (1 == det.unique()):
            det = "SBND"
        else:
            det = "ICARUS"
        #check ismc
        hdrdf = make_mchdrdf(f)
        ismc = hdrdf.ismc.iloc[0]

        chi2_pids = []
        for plane in range(0, 3):
            trkhitdf = make_trkhitdf(f, plane)
            #if det == "SBND": ## FIXME
            #    trkhitdf = trkhitdf[InFV(df = trkhitdf, inzback = 0., det = "SBND_nohighyz")]
            #dqdx_redo = chi2pid.dqdx(trkhitdf, gain=None, calibrate=det, isMC=ismc)
            dedx_redo = chi2pid.dedx(trkhitdf, gain=det, calibrate=det, plane=plane, isMC=ismc)
            #dedx_bias = (dedx_redo - trkhitdf.dedx) / trkhitdf.dedx
            trkhitdf["dedx_redo"] = dedx_redo
            #trkhitdf["dqdx_redo"] = dqdx_redo
            #trkhitdf["dedx_bias"] = dedx_bias
            #trkhitdf["integ_ov_pitch"] = trkhitdf.integral / trkhitdf.pitch
            #print(trkhitdf[trkhitdf.rr < 26.][['dedx', 'dedx_redo', 'dedx_bias', 'tpc', 'run', 'iov', 'rho']].head(50))
            #if plane == 2:
            #    print(trkhitdf[trkhitdf.rr < 26.].loc[(3, 0, 0), ['dedx', 'dedx_redo', 'dedx_bias', 'dqdx', 'dqdx_redo', 'etau_corr', 'yz_scale', 'integ_ov_pitch', 'integral', 'pitch', 'tpc', 'x', 'y', 'z', 'run', 'iov', 'rho', 'rr']])
            #    print(trkhitdf[trkhitdf.rr < 26.].loc[(3, 0, 1), ['dedx', 'dedx_redo', 'dedx_bias', 'dqdx', 'dqdx_redo', 'etau_corr', 'yz_scale', 'integ_ov_pitch', 'integral', 'pitch', 'tpc', 'x', 'y', 'z', 'run', 'iov', 'rho', 'rr']])
            #    print(trkhitdf[trkhitdf.rr < 26.].loc[(4, 0, 1), ['dedx', 'dedx_redo', 'dedx_bias', 'dqdx', 'dqdx_redo', 'etau_corr', 'yz_scale', 'integ_ov_pitch', 'integral', 'pitch', 'tpc', 'x', 'y', 'z', 'run', 'iov', 'rho', 'rr']])
            #    print(trkhitdf[trkhitdf.rr < 26.].loc[(5, 1, 0), ['dedx', 'dedx_redo', 'dedx_bias', 'dqdx', 'dqdx_redo', 'etau_corr', 'yz_scale', 'integ_ov_pitch', 'integral', 'pitch', 'tpc', 'x', 'y', 'z', 'run', 'iov', 'rho', 'rr']])
            for par in ['muon', 'proton', 'pion']:
                this_chi2_new, this_chi2_ndof = chi2pid.chi2par(trkhitdf, dedxname="dedx_redo", par=par)
                this_chi2_col = ('pfp', 'trk', 'chi2pid', 'I' + str(plane), 'chi2_' + par + '_new', '')
                this_ndof_col = ('pfp', 'trk', 'chi2pid', 'I' + str(plane), 'ndof_' + par + '_new', '')
                trkdf[this_chi2_col] = this_chi2_new
                trkdf[this_ndof_col] = this_chi2_ndof
                trkdf[this_chi2_col] = trkdf[this_chi2_col].fillna(-999)
                trkdf[this_ndof_col] = trkdf[this_ndof_col].fillna(-999)

    slcdf = make_slcdf(f)

    # merge in tracks
    slcdf = multicol_merge(slcdf, trkdf, left_index=True, right_index=True, how="right", validate="one_to_many")
    #print(slcdf[slcdf.slc.is_clear_cosmic==0].pfp.trk.chi2pid.I2.head(30))
    #print(slcdf[slcdf.slc.is_clear_cosmic==0].pfp.trackScore.head(30))

    # distance from vertex to track start
    slcdf = multicol_add(slcdf, dmagdf(slcdf.slc.vertex, slcdf.pfp.trk.start).rename(("pfp", "dist_to_vertex")))

    if trkDistCut > 0:
        slcdf = slcdf[slcdf.pfp.dist_to_vertex < trkDistCut]
    if cutClearCosmic:
        slcdf = slcdf[slcdf.slc.is_clear_cosmic==0]
    # require fiducial verex
    if requireFiducial:
        slcdf = slcdf[InFV(slcdf.slc.vertex, 50)]

    #print(slcdf.pfp.trk.chi2pid.head(50))
    return slcdf

def make_stubs(f, det="ICARUS"):
    stubdf = loadbranches(f["recTree"], stubbranches)
    stubdf = stubdf.rec.slc.reco.stub

    stubpdf = loadbranches(f["recTree"], stubplanebranches)
    stubpdf = stubpdf.rec.slc.reco.stub.planes

    stubdf["nplane"] = stubpdf.groupby(level=[0,1,2]).size()
    stubdf["plane"] = stubpdf.p.groupby(level=[0,1,2]).first()

    stubhitdf = loadbranches(f["recTree"], stubhitbranches)
    stubhitdf = stubhitdf.rec.slc.reco.stub.planes.hits

    stubhitdf = stubhitdf.join(stubpdf)
    stubhitdf = stubhitdf.join(stubdf.efield_vtx)
    stubhitdf = stubhitdf.join(stubdf.efield_end)

    hdrdf = make_mchdrdf(f)
    def dEdx2dQdx(dEdx): # MC parameters
        return recombination_sbnd(dEdx, np.pi/2) if det == "SBND" else recombination_icarus(dEdx, np.pi/2)

    MIP_dqdx = dEdx2dQdx(1.7) 

    stub_end_charge = stubhitdf.charge[stubhitdf.wire == stubhitdf.hit_w].groupby(level=[0,1,2,3]).first().groupby(level=[0,1,2]).first()
    stub_end_charge.name = ("endp_charge", "", "")

    stub_pitch = stubpdf.pitch.groupby(level=[0,1,2]).first()
    stub_pitch.name = ("pitch", "", "")

    stubdir_is_pos = (stubhitdf.hit_w - stubhitdf.vtx_w) > 0.
    when_sum = ((stubhitdf.wire > stubhitdf.vtx_w) == stubdir_is_pos) & (((stubhitdf.wire < stubhitdf.hit_w) == stubdir_is_pos) | (stubhitdf.wire == stubhitdf.hit_w)) 
    stubcharge = (stubhitdf.charge[when_sum]).groupby(level=[0,1,2,3]).sum().groupby(level=[0,1,2]).first()
    stubcharge.name = ("charge", "", "")

    stubinccharge = (stubhitdf.charge).groupby(level=[0,1,2,3]).sum().groupby(level=[0,1,2]).first()
    stubinccharge.name = ("inc_charge", "", "")

    hit_before_start = ((stubhitdf.wire < stubhitdf.vtx_w) == stubdir_is_pos)
    stub_inc_sub_charge = (stubhitdf.charge - MIP_dqdx*stubhitdf.ontrack*(~hit_before_start)*stubhitdf.trkpitch).groupby(level=[0,1,2,3]).sum().groupby(level=[0,1,2]).first()
    stub_inc_sub_charge.name = ("inc_sub_charge", "", "")

    stubdf = stubdf.join(stubcharge)
    stubdf = stubdf.join(stubinccharge)
    stubdf = stubdf.join(stub_inc_sub_charge)
    stubdf = stubdf.join(stub_end_charge)
    stubdf = stubdf.join(stub_pitch)
    stubdf["length"] = magdf(stubdf.vtx - stubdf.end)
    stubdf["Q"] = stubdf.inc_sub_charge
    stubdf["truth_pdg"] = stubdf.truth.p.pdg
    stubdf["truth_interaction_id"] = stubdf.truth.p.interaction_id 
    stubdf["truth_gen_E"] = stubdf.truth.p.genE 

    # TODO: convert charge to energy
    stubdf["ke"] = np.nan # Q2KE(stubdf.Q)
    # TODO: also do calorimetric variations
    stubdf["ke_callo"] = np.nan # Q2KE_mc_callo(stubdf.Q)
    stubdf["ke_calhi"] = np.nan # Q2KE_mc_calhi(stubdf.Q)

    stubdf.ke = stubdf.ke.fillna(0)
    stubdf.Q = stubdf.Q.fillna(0)

    stubdf["dedx"] = stubdf.ke / stubdf.length

    stubdf["dedx_callo"] = stubdf.ke_callo / stubdf.length
    stubdf["dedx_calhi"] = stubdf.ke_calhi / stubdf.length

    dqdx = stubdf.inc_sub_charge / stubdf.length
    length = stubdf.length
    hasstub = (length < 4.) & \
        (((length > 0.) & (dqdx > 5.5e5)) |\
        ((length > 0.5) & (dqdx > 3.5e5)) |\
        ((length > 1) & (dqdx > 3e5)) |\
        ((length > 2) & (dqdx > 2e5)))

    stubdf["dqdx"] = dqdx 
    stubdf['pass_proton_stub'] = hasstub
    return stubdf

    ## It seems there is a bug. First stub in each length is included for a slice...
    # only take collection plane
    #stubdf = stubdf[stubdf.plane == 2]

    #stub_length_bins = [0, 0.5, 1, 2, 3, 4]
    #stub_length_name = ["l0_5cm", "l1cm", "l2cm", "l3cm", "l4cm"]
    #tosave = ["dedx", "dedx_callo", "dedx_calhi", "Q", "length", "charge", "inc_charge"]

    #df_tosave = []
    #for blo, bhi, name in zip(stub_length_bins[:-1], stub_length_bins[1:], stub_length_name):
    #    stub_tosave = stubdf.dedx[(stubdf.length > blo) & (stubdf.length < bhi)].groupby(level=[0,1]).idxmax()
    #    for col in tosave:
    #        s = stubdf.loc[stub_tosave, col]
    #        s.name = ("stub", name, col, "", "", "")
    #        s.index = s.index.droplevel(-1)
    #        df_tosave.append(s)

    #return pd.concat(df_tosave, axis=1)

def make_all_spine_df(f, get_best_match=True, **trkArgs):
    # Load SPINE interactions dataframe
    spine_int_df = make_spine_int_df(f, get_best_match)

    # Load SPINE particles dataframe
    spine_part_df = make_spine_part_df(f, get_best_match, **trkArgs)

    # Merge both dataframes
    spine_df = multicol_merge(spine_int_df, spine_part_df, left_index=True, right_index=True, how="right", validate="one_to_many")

    return spine_df

def make_spine_int_mcnu_df(f, get_best_match=True):
    """
    SPINE dataframe maker which containes all SPINE reco and true interaction variables 
    matched with MC nu dataframe.
    
    :param f: file handle to the input ROOT file
    :type f: uproot4.open file handle

    :return: SPINE + MC nu events dataframe
    :rtype: pd.DataFrame   
    """

    # Load SPINE and MC dataframes.
    spineint_df = make_spine_int_df(f, get_best_match)
    mcnu_df = make_mcdf(f)

    # Match reco and true SPINE interactions with MC interactions data.
    add_upper_level_to_df("mcnu", mcnu_df)
    spineint_mcnu_df = multicol_merge(
        lhs=spineint_df,
        rhs=mcnu_df,
        left_on=["entry", ("rec", "dlp_true", "mct_index", "")],
        right_index=True,
        how="left",
        validate="many_to_one"
    )

    return spineint_mcnu_df

def make_spine_part_mcpart_df(f, get_best_match=True):
    """
    SPINE dataframe maker which containes all SPINE reco and true particle variables 
    matched with MC part dataframe.
    
    :param f: file handle to the input ROOT file
    :type f: uproot4.open file handle

    :return: SPINE + MC particles dataframe
    :rtype: pd.DataFrame   
    """

    # Load SPINE particles dataframe.
    spinepart_df = make_spine_part_df(f, get_best_match)

    # Load MC (Geant4) particles branches.
    mcpart_df = loadbranches(f["recTree"], trueparticlebranches)

    # Match SPINE particles dataframe with MC (Geant4) particles dataframe.
    spinepart_mcpart_df = multicol_merge(
        lhs=spinepart_df.reset_index(level=[1, 2]),
        rhs=mcpart_df.reset_index(level=[1]),
        left_on=["entry", ("rec", "dlp_true", "particles", "track_id", "")],
        right_on=["entry", ("rec", "true_particles", "G4ID", "", "", "")],
        how="left",
        validate="many_to_one"
    ).set_index(["rec.dlp..index", "rec.dlp.particles..index"], append=True)

    return spinepart_mcpart_df

def make_spine_int_df(f, get_best_match=True):
    """
    SPINE dataframe maker which containes all SPINE reco and true interaction variables
    matched with MC nu dataframe.
    
    :param f: file handle to the input ROOT file
    :type f: uproot4.open file handle
    :param get_best_match: `True` if a best match selection is needed.
    :type get_best_match: bool

    :return: SPINE interactions dataframe
    :rtype: pd.DataFrame   
    """
    # Get reco tree from file.
    rec_tree = f["recTree"]

    # Load SPINE interaction and matches branches.
    spinetint_df            = loadbranches(rec_tree, spinetint_branches)
    spineint_df             = loadbranches(rec_tree, spineint_branches)
    spineint_matchids_df    = loadbranches(rec_tree, spineint_matched_branches)
    spineint_matchovrlp_df  = loadbranches(rec_tree, spineint_matchovrlp_branches)

    # Match match_ids and match_overlaps data.
    spineint_matches_df = spineint_matchids_df.merge(
        spineint_matchovrlp_df,
        left_on=["entry", "rec.dlp..index", "rec.dlp.match_ids..index"],
        right_on=["entry", "rec.dlp..index", "rec.dlp.match_overlaps..index"],
        how="left",
        validate="one_to_one"
    )

    # If we want to get the best match, first sorts the match_overlaps column and gets the first value.
    # matches_df, validate_matched and validate_matched_with_true values change depending on get_best_match value.
    matches_df = spineint_matches_df
    validate_matched = "many_to_one"
    validate_matched_with_true = "many_to_many"
    if get_best_match:
        # Find best match for each Multi-Index.
        spineint_best_matches_df = (
            spineint_matches_df
            .sort_values(("rec", "dlp", "match_overlaps"), ascending=False)
            .groupby(level=["entry", "rec.dlp..index"])
            .first()
        )
        matches_df = spineint_best_matches_df
        validate_matched = "one_to_one"
        validate_matched_with_true = "many_to_one"

    # Match reco SPINE interactions to match_ids and match_overlaps data.
    spineint_matched_df = multicol_merge(
        lhs=matches_df,
        rhs=spineint_df,
        on=["entry", "rec.dlp..index"],
        how="left",
        validate=validate_matched
    )

    # Match reco SPINE interactions with true SPINE interactions.
    spineint_matched_with_true_df = multicol_merge(
        lhs=spineint_matched_df.reset_index(level=1),
        rhs=spinetint_df.reset_index(level=1),
        left_on=["entry", ("rec", "dlp", "match_ids", "")],
        right_on=["entry", ("rec", "dlp_true", "id", "")],
        how="left",
        validate=validate_matched_with_true
    ).set_index(("rec.dlp..index"), append=True)

    # Rename I0-I1-I2 variables to x-y-z
    cols_to_rename = ["momentum", "end_point", "start_point", "start_dir", "end_dir", "reco_vertex", "vertex"]
    rename_to_XYZ(spineint_matched_with_true_df, cols_to_rename)

    return spineint_matched_with_true_df

def make_spine_part_df(f, get_best_match=True):
    """
    SPINE dataframe maker which containes all SPINE reco and true particle variables 
    matched with MC nu dataframe.
    
    :param f: file handle to the input ROOT file
    :type f: uproot4.open file handle
    :param get_best_match: `True` if a best match selection is needed.
    :type get_best_match: bool

    :return: SPINE particles dataframe
    :rtype: pd.DataFrame   
    """
    rec_tree = f["recTree"]

    spinepart_df                = loadbranches(rec_tree, spinepart_branches)
    spinetpart_df               = loadbranches(rec_tree, spinetpart_branches)
    spinepart_matchids_df       = loadbranches(rec_tree, spinepart_matched_branches)
    spinepart_matchovrlp_df     = loadbranches(rec_tree, spinepart_matchovrlp_branches)

    # Match match_ids and match_overlaps data.
    spinepart_matches_df = spinepart_matchids_df.merge(
        spinepart_matchovrlp_df,
        left_on = ["entry", "rec.dlp..index", "rec.dlp.particles..index", "rec.dlp.particles.match_ids..index"],
        right_on = ["entry", "rec.dlp..index", "rec.dlp.particles..index", "rec.dlp.particles.match_overlaps..index"],
        how="left",
        validate="one_to_one"
    )

    # If we want to get the best match, first sorts the match_overlaps column and gets the first value.
    # matches_df, validate_matched and validate_matched_with_true values change depending on get_best_match value.
    matches_df = spinepart_matches_df
    validate_matched = "many_to_one"
    validate_matched_with_true = "many_to_many"
    if get_best_match:
        # Get best interaction matching
        spinepart_best_matches_df = (
            spinepart_matches_df
            .sort_values(("rec", "dlp", "particles", "match_overlaps"), ascending=False)
            .groupby(level=["entry", "rec.dlp..index", "rec.dlp.particles..index"])
            .first()
        )
        matches_df = spinepart_best_matches_df
        validate_matched = "one_to_one"
        validate_matched_with_true = "many_to_one"

    # Match reco SPINE particles to match_ids and match_overlaps data.
    spinepart_matched_df = multicol_merge(
        lhs=matches_df,
        rhs=spinepart_df,
        on=["entry", "rec.dlp..index", "rec.dlp.particles..index"],
        how="left",
        validate=validate_matched
    )

    # Match reco SPINE particles with true SPINE particles.
    spinepart_matched_with_true_df = multicol_merge(
        lhs=spinepart_matched_df.reset_index(level=[1, 2]),
        rhs=spinetpart_df.reset_index(level=[1, 2]),
        left_on=["entry", ("rec", "dlp", "particles", "match_ids")],
        right_on=["entry", ("rec", "dlp_true", "particles", "id")],
        how="left",
        validate=validate_matched_with_true
    ).set_index(["rec.dlp..index", "rec.dlp.particles..index"], append=True)

    # Rename I0-I1-I2 variables to x-y-z
    cols_to_rename = ["momentum", "end_point", "start_point", "start_dir", "end_dir", "vertex"] 
    rename_to_XYZ(spinepart_matched_with_true_df, cols_to_rename)

    return spinepart_matched_with_true_df

def make_spine_flash_df(f):
    """
    SPINE dataframe maker which containes flash branches from true and reco SPINE interactions.

    :param f: file handle to the input ROOT file
    :type f: uproot4.open file handle

    :return: SPINE flashes dataframe
    :rtype: pd.DataFrame   
    """

    rec_tree = f["recTree"]

    # Load SPINE flash branches for reco interactions.
    spineint_flashids_df        = loadbranches(rec_tree, spineint_flashids_branches)
    spineint_flashscores_df     = loadbranches(rec_tree, spineint_flashscores_branches)
    spineint_flashtimes_df      = loadbranches(rec_tree, spineint_flashtimes_branches)
    spineint_flashvolumeids_df  = loadbranches(rec_tree, spineint_flashvolumeids_branches)

    # Load SPINE flash branches for true interactions.
    spinetint_flashids_df       = loadbranches(rec_tree, spinetint_flashids_branches)
    spinetint_flashscores_df    = loadbranches(rec_tree, spinetint_flashscores_branches)
    spinetint_flashtimes_df     = loadbranches(rec_tree, spinetint_flashtimes_branches)
    spinetint_flashvolumeids_df = loadbranches(rec_tree, spinetint_flashvolumeids_branches)

    # Unify column index names across all flash dataframes.
    dfs = [spineint_flashids_df, spineint_flashscores_df, spineint_flashtimes_df, spineint_flashvolumeids_df,
        spinetint_flashids_df, spinetint_flashscores_df, spinetint_flashtimes_df, spinetint_flashvolumeids_df]
    for df in dfs:
        assert df.index.nlevels == 3, f"Unexpected index depth: {df.index.nlevels}"
        df.index.names = ["entry", "rec.dlp..index", "rec.dlp.flash..index"]

    # Join reco and true flash dataframes.
    spine_flash_df = reduce(lambda l, r: l.join(r, how='outer'), dfs)

    return spine_flash_df

def make_corrections_df(f):
    corE = loadbranches(f["recTree"], corrections)
    return corE


#InFV_SBND = functools.partial(util.InFV, inzback=np.nan, det='SBND')
#InFV_IC = functools.partial(util.InFV, inzback=np.nan, det='ICARUS')

def make_muNp0pidf(f: pd.DataFrame, save_track_truth: bool=False) -> pd.DataFrame:
    pandora_df = make_pandora_df(f, trkDistCut=0)

    # precuts
    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    run = loadbranches(f["recTree"], ["rec.hdr.run"]).rec.hdr.run

    pandora_df = pandora_df[pandora_df.slc.is_clear_cosmic == 0]

    # daughter info
    '''
    rec
                                                                                             slc
                                                                                            reco
                                                                                             pfp
                                                                                       daughters
        entry rec.slc..index rec.slc.reco.pfp..index rec.slc.reco.pfp.daughters..index
        0     0              1                       0                                         3
              1              1                       0                                         6
              2              3                       0                                        14
                                                     1                                        27
                                                     2                                        38
    daughter_to_pfp = daughterdf.reset_index().set_index(['entry', 'rec.slc..index', ('rec', 'slc', 'reco', 'pfp', 'daughters')])
    daughter_df = pandora_df[pandora_df.index.isin(daughter_to_pfp.index)]
    print(daughter_df.pfp)
    '''

    daughterdf = ph.loadbranches(f["recTree"], branches.pfp_daughter_branch)
    ndaughterdf = daughterdf.groupby(level=[0, 1, 2]).count()
    daughter_id = (
            daughterdf[
                daughterdf.index.isin(ndaughterdf[ndaughterdf.rec.slc.reco.pfp.daughters == 1].index)
            ].reset_index(level=-1)
    )[[('rec', 'slc', 'reco', 'pfp', 'daughters')]].droplevel([0, 1, 2], axis='columns')

    pandora_df = ph.multicol_merge(pandora_df, daughter_id, left_index=True, right_index=True, how='left')

    # drop things we don't need
    mask = (
        (
            (pandora_df.columns.get_level_values(0) == "pfp")
            & (pandora_df.columns.get_level_values(1).isin(
                ["trk", "trackScore", "dist_to_vertex", "t0", "daughters"]
            ) | (save_track_truth & (pandora_df.columns.get_level_values(2) == "truth")))
        )
        | (
            (pandora_df.columns.get_level_values(0) == "slc")
            & (pandora_df.columns.get_level_values(1).isin(
                ["tmatch", "vertex", "bcfm_score", "nu_score"]
            ) | ((pandora_df.columns.get_level_values(1) == "truth")
                &
                (pandora_df.columns.get_level_values(2).isin(
                    ["pdg", "E", "baseline", "genie_mode", "q0_lab", "modq_lab", "Q2"]
                )))
        )
    )
    )
    pandora_df = pandora_df.loc[:, mask]

    #recodf = all_fv_cuts(pandora_df, "SBND")
    recodf = pandora_df
    recodf[('pfp', 'trk', 'particle_reco', '', '', '')] = np.zeros(len(recodf))

    recodf = multicol_add(recodf, (np.abs(recodf.pfp.trk.truth.p.pdg)==2112).groupby(level=[0,1,2]).sum().rename(('slc', 'truth', 'nn')))
    recodf = multicol_add(recodf, (np.abs(recodf.pfp.trk.truth.p.pdg)==2212).groupby(level=[0,1,2]).sum().rename(('slc', 'truth', 'np')))
    recodf = multicol_add(recodf, (np.abs(recodf.pfp.trk.truth.p.pdg)==13).groupby(level=[0,1,2]).sum().rename(('slc', 'truth', 'nmu')))
    recodf = multicol_add(recodf, (np.abs(recodf.pfp.trk.truth.p.pdg)==211).groupby(level=[0,1,2]).sum().rename(('slc', 'truth', 'npi')))

    recodf = muNp0pi_pipeline(recodf, 30, 80, 25)

    return recodf

    

