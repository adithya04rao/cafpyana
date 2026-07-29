# Standard library imports
import os
import sys

# Third-party imports
import pandas as pd
import numpy as np

# Add the head direcoty to sys.path
workspace_root = os.getcwd()
sys.path.insert(0, workspace_root + "/../../")

# Local imports
import analysis_village.gump.kinematics
from makedf.util import *

# Fiducial volume cuts for SBND and ICARUS
SBNDFVCuts = {
    "lowYZ": {
        "x": {"min": -200., "max": 200.},
        "y": {"min": -200., "max": 200.},
        "z": {"min": 0., "max": 250.}
    },
    "highYZ": {
        "x": {"min": -200., "max": 200.},
        "y": {"min": -200., "max": 200},
        "z": {"min": 250., "max": 500.}
    }
}

ICARUSRun2FVCuts = {
    "C0": {
        "x": {"min": -210.22, "max": -61.94}, # exluce EE in Run 2
        "y": {"min": -181.86, "max": 134.96},
        "z": {"min": -894.950652270838, "max": 894.950652270838}
    },
    "C1": {
        "x": {"min": 61.94, "max": 358.49},
        "y": {"min": -181.86, "max": 134.96},
        "z": {"min": -894.950652270838, "max": 894.950652270838}
    }
}

ICARUSRun4FVCuts = {
    "C0": {
        "x": {"min": -358.49, "max": -61.94},
        "y": {"min": -181.86, "max": 134.96},
        "z": {"min": -894.950652270838, "max": 894.950652270838}
    },
    "C1": {
        "x": {"min": 61.94, "max": 358.49},
        "y": {"min": -181.86, "max": 134.96},
        "z": {"min": -894.950652270838, "max": 894.950652270838}
    }
}

"""def _fv_cut_slc(df, det, inx=10, iny=10, inzfront=10, inzback=50):
    if det == "SBND":
        return ((df.slc.vertex.x < SBNDFVCuts['lowYZ']['x']['max'] - inx) & (df.slc.vertex.x > SBNDFVCuts['lowYZ']['x']['min'] + inx) &\
                (df.slc.vertex.y < SBNDFVCuts['lowYZ']['y']['max'] - iny) & (df.slc.vertex.y > SBNDFVCuts['lowYZ']['y']['min'] + iny) &\
                (df.slc.vertex.z < SBNDFVCuts['lowYZ']['z']['max']) & (df.slc.vertex.z > SBNDFVCuts['lowYZ']['z']['min'] + inzfront)) |\
               ((df.slc.vertex.x < SBNDFVCuts['highYZ']['x']['max'] - inx) & (df.slc.vertex.x > SBNDFVCuts['highYZ']['x']['min'] + inx) &\
                (df.slc.vertex.y < SBNDFVCuts['highYZ']['y']['max'] - iny) & (df.slc.vertex.y > SBNDFVCuts['highYZ']['y']['min'] + iny) &\
                (df.slc.vertex.z < SBNDFVCuts['highYZ']['z']['max'] - inzback) & (df.slc.vertex.z > SBNDFVCuts['highYZ']['z']['min']))

    else:
        raise NameError("DETECTOR not valid, should be SBND or ICARUS Run2 or ICARUS Run4")
    
def _fv_cut_trk(df, det, inx=10, iny=10, inzfront=10, inzback=50):
    if det == "SBND":
        return ((df.pfp.trk.start.x < SBNDFVCuts['lowYZ']['x']['max'] - inx) & (df.pfp.trk.start.x > SBNDFVCuts['lowYZ']['x']['min'] + inx) &\
                (df.pfp.trk.start.y < SBNDFVCuts['lowYZ']['y']['max'] - iny) & (df.pfp.trk.start.y > SBNDFVCuts['lowYZ']['y']['min'] + iny) &\
                (df.pfp.trk.start.z < SBNDFVCuts['lowYZ']['z']['max']) & (df.pfp.trk.start.z > SBNDFVCuts['lowYZ']['z']['min'] + inzfront)) |\
               ((df.pfp.trk.start.x < SBNDFVCuts['highYZ']['x']['max'] - inx) & (df.pfp.trk.start.x > SBNDFVCuts['highYZ']['x']['min'] + inx) &\
                (df.pfp.trk.start.y < SBNDFVCuts['highYZ']['y']['max'] - iny) & (df.pfp.trk.start.y > SBNDFVCuts['highYZ']['y']['min'] + iny) &\
                (df.pfp.trk.start.z < SBNDFVCuts['highYZ']['z']['max'] - inzback) & (df.pfp.trk.start.z > SBNDFVCuts['highYZ']['z']['min'])) &\
                ((df.pfp.trk.end.x < SBNDFVCuts['lowYZ']['x']['max'] - inx) & (df.pfp.trk.end.x > SBNDFVCuts['lowYZ']['x']['min'] + inx) &\
                (df.pfp.trk.end.y < SBNDFVCuts['lowYZ']['y']['max'] - iny) & (df.pfp.trk.end.y > SBNDFVCuts['lowYZ']['y']['min'] + iny) &\
                (df.pfp.trk.end.z < SBNDFVCuts['lowYZ']['z']['max']) & (df.pfp.trk.end.z > SBNDFVCuts['lowYZ']['z']['min'] + inzfront)) |\
               ((df.pfp.trk.end.x < SBNDFVCuts['highYZ']['x']['max'] - inx) & (df.pfp.trk.end.x > SBNDFVCuts['highYZ']['x']['min'] + inx) &\
                (df.pfp.trk.end.y < SBNDFVCuts['highYZ']['y']['max'] - iny) & (df.pfp.trk.end.y > SBNDFVCuts['highYZ']['y']['min'] + iny) &\
                (df.pfp.trk.end.z < SBNDFVCuts['highYZ']['z']['max'] - inzback) & (df.pfp.trk.end.z > SBNDFVCuts['highYZ']['z']['min'])) &\
                (df.pfp.trk.start.x.notna())
    elif "ICARUS" in det:
            FVRun2 = (((df.pfp.trk.end.x < (ICARUSRun2FVCuts['C0']['x']['max'] - inx)) & (df.pfp.trk.end.x > (ICARUSRun2FVCuts['C0']['x']['min'] + inx))) |\
                    ((df.pfp.trk.end.x < (ICARUSRun2FVCuts['C1']['x']['max'] - inx)) & (df.pfp.trk.end.x > (ICARUSRun2FVCuts['C1']['x']['min'] + inx)))) &\
                     (df.pfp.trk.end.y < (ICARUSRun2FVCuts['C0']['y']['max'] - iny)) & (df.pfp.trk.end.y > (ICARUSRun2FVCuts['C0']['y']['min'] + iny)) &\
                     (df.pfp.trk.end.z < (ICARUSRun2FVCuts['C0']['z']['max'] - inzback)) & (df.pfp.trk.end.z > (ICARUSRun2FVCuts['C0']['z']['min'] + inzfront))
            FVRun4 = (((df.pfp.trk.end.x < (ICARUSRun4FVCuts['C0']['x']['max'] - inx)) & (df.pfp.trk.end.x > (ICARUSRun4FVCuts['C0']['x']['min'] + inx))) |\
                    ((df.pfp.trk.end.x < (ICARUSRun4FVCuts['C1']['x']['max'] - inx)) & (df.pfp.trk.end.x > (ICARUSRun4FVCuts['C1']['x']['min'] + inx)))) &\
                     (df.pfp.trk.end.y < (ICARUSRun4FVCuts['C0']['y']['max'] - iny)) & (df.pfp.trk.end.y > (ICARUSRun4FVCuts['C0']['y']['min'] + iny)) &\
                     (df.pfp.trk.end.z < (ICARUSRun4FVCuts['C0']['z']['max'] - inzback)) & (df.pfp.trk.end.z > (ICARUSRun4FVCuts['C0']['z']['min'] + inzfront))
            if det == "ICARUS":
                ret = FVRun2
                ret[df.Run == 4] = FVRun4[df.Run == 4]
                return ret
            elif det == "ICARUS Run2":
                return FVRun2
            elif det == "ICARUS Run4":
                return FVRun4
            else:
                raise NameError("DETECTOR not valid, should be SBND or ICARUS Run2 or ICARUS Run4")
    else:
        raise NameError("DETECTOR not valid, should be SBND or ICARUS Run2 or ICARUS Run4")"""

def _fv_cut(df, det, inx=10, iny=10, inzfront=10, inzback=50):
    if "ICARUS" in det:
        FVRun2 = (((df.x < (ICARUSRun2FVCuts['C0']['x']['max'] - inx)) & (df.x > (ICARUSRun2FVCuts['C0']['x']['min'] + inx))) |\
                ((df.x < (ICARUSRun2FVCuts['C1']['x']['max'] - inx)) & (df.x > (ICARUSRun2FVCuts['C1']['x']['min'] + inx)))) &\
                 (df.y < (ICARUSRun2FVCuts['C0']['y']['max'] - iny)) & (df.y > (ICARUSRun2FVCuts['C0']['y']['min'] + iny)) &\
                 (df.z < (ICARUSRun2FVCuts['C0']['z']['max'] - inzback)) & (df.z > (ICARUSRun2FVCuts['C0']['z']['min'] + inzfront))
        FVRun4 = (((df.x < (ICARUSRun4FVCuts['C0']['x']['max'] - inx)) & (df.x > (ICARUSRun4FVCuts['C0']['x']['min'] + inx))) |\
                ((df.x < (ICARUSRun4FVCuts['C1']['x']['max'] - inx)) & (df.x > (ICARUSRun4FVCuts['C1']['x']['min'] + inx)))) &\
                 (df.y < (ICARUSRun4FVCuts['C0']['y']['max'] - iny)) & (df.y > (ICARUSRun4FVCuts['C0']['y']['min'] + iny)) &\
                 (df.z < (ICARUSRun4FVCuts['C0']['z']['max'] - inzback)) & (df.z > (ICARUSRun4FVCuts['C0']['z']['min'] + inzfront))
        if det == "ICARUS":
            ret = FVRun2
            ret[df.Run == 4] = FVRun4[df.Run == 4]
            return ret
        elif det == "ICARUS Run2":
            return FVRun2
        elif det == "ICARUS Run4":
            return FVRun4
        else:
            raise NameError("DETECTOR not valid, should be SBND or ICARUS Run2 or ICARUS Run4")
    elif det == "SBND":
        return ((df.x < SBNDFVCuts['lowYZ']['x']['max'] - inx) & (df.x > SBNDFVCuts['lowYZ']['x']['min'] + inx) &\
                (df.y < SBNDFVCuts['lowYZ']['y']['max'] - iny) & (df.y > SBNDFVCuts['lowYZ']['y']['min'] + iny) &\
                (df.z < SBNDFVCuts['lowYZ']['z']['max']) & (df.z > SBNDFVCuts['lowYZ']['z']['min'] + inzfront)) |\
               ((df.x < SBNDFVCuts['highYZ']['x']['max'] - inx) & (df.x > SBNDFVCuts['highYZ']['x']['min'] + inx) &\
                (df.y < SBNDFVCuts['highYZ']['y']['max'] - iny) & (df.y > SBNDFVCuts['highYZ']['y']['min'] + iny) &\
                (df.z < SBNDFVCuts['highYZ']['z']['max'] - inzback) & (df.z > SBNDFVCuts['highYZ']['z']['min']))

    else:
        raise NameError("DETECTOR not valid, should be SBND or ICARUS Run2 or ICARUS Run4")

def slcfv_cut(df, det):
    vtx = pd.DataFrame({
                           'Run': df.Run,
                           'x': df.slc.vertex.x,
                           'y': df.slc.vertex.y,
                           'z': df.slc.vertex.z}, index=df.index)
    return _fv_cut(vtx, det)

def trkfv_cut(df, det):
    trk = pd.DataFrame({
                               'Run': df.Run,
                               'x': df.pfp.trk.end.x,
                               'y': df.pfp.trk.end.y,
                               'z': df.pfp.trk.end.z}, index=df.index)
    return _fv_cut(trk, det, inzback=10)

def cosmic_cut(df):
    return (df.slc.nu_score > 0.4)

def crthitveto_cut(df):
    return df.pfp.trk.crthit.hit.pe.isna()

mode_list = [0, 10, 1, 2, 3]
mode_labels = ['QE', 'MEC', 'RES', 'SIS/DIS', 'COH', "other"]

def breakdown_mode(var, df):
    """Break down variable by interaction mode."""
    ret = [var[df.genie_mode == i] for i in mode_list]
    ret.append(var[sum([df.genie_mode == i for i in mode_list]) == 0])
    return ret


def all_fv_cuts(recodf, DETECTOR):

    ###Random
    recodf = recodf[recodf.pfp.trk.start.x.notna()]

    ### Slice containment cut
    recodf = recodf[slcfv_cut(recodf, DETECTOR)]

    ### Track containment cut (require that all tracks within a slice are contained)
    recodf = recodf.loc[trkfv_cut(recodf, DETECTOR).groupby(level=[0,1,2]).transform("all")]

    ### NuScore cut
    recodf = recodf[cosmic_cut(recodf)]

    ### crthitveto cut
    if "ICARUS" in DETECTOR:
        recodf = recodf[crthitveto_cut(recodf)]

    return recodf

def chi2_pid_correction(df):
    mask1 = ((df.pfp.trk.truth.p.pdg != 2212) & (df.pfp.trk.chi2pid.I2.chi2_proton == 0))                                     #non protons with chi2_proton = 0 
    mask2 = ((df.pfp.trk.truth.p.pdg != 13) & (df.pfp.trk.truth.p.pdg != -13) & (df.pfp.trk.chi2pid.I2.chi2_muon == 0))       #non muons with chi2_muon = 0
    mask3 = ((df.pfp.trk.truth.p.pdg != 211) & (df.pfp.trk.truth.p.pdg != -211) & (df.pfp.trk.chi2pid.I2.chi2_muon == 0))     #non pions with chi2_pion = 0

    df.loc[mask1, ('pfp','trk','chi2pid','I2','chi2_proton','')] = -999
    df.loc[mask2, ('pfp','trk','chi2pid','I2','chi2_muon','')] = -999
    df.loc[mask3, ('pfp','trk','chi2pid','I2','chi2_pion','')] = -999

    return df

def particle_classification(df, cut_m, cut_p, muon_len_cut):
    import warnings
    warnings.filterwarnings("ignore")

    mask = ~df[('pfp', 'trk', 'truth', 'p', 'pdg', '')].isin([13,-13,2212])
    df[('slc', 'truth', 'n_oth', '', '', '')] = mask.groupby(level=[0,1,2]).transform('sum')

    muons = ((df[('pfp', 'trk', 'chi2pid', 'I2', 'chi2_muon')] < cut_m) & (df[('pfp', 'trk', 'chi2pid', 'I2', 'chi2_proton')] > cut_p) & (df[('pfp', 'trk', 'len')] > muon_len_cut))
    protons = ((df[('pfp', 'trk', 'chi2pid', 'I2', 'chi2_muon')] > cut_m) & (df[('pfp', 'trk', 'chi2pid', 'I2', 'chi2_proton')] < cut_p))
    notnan = (df[('pfp', 'trk', 'truth', 'p', 'pdg')].notna())

    df.loc[muons, ('pfp', 'trk', 'particle_reco', '', '', '')] = 'muon'
    df.loc[protons, ('pfp', 'trk', 'particle_reco', '', '', '')] = 'proton'
    df.loc[~(muons | protons) & notnan, ('pfp', 'trk', 'particle_reco', '', '', '')] = 'other'

    mask = df[('pfp', 'trk', 'particle_reco', '', '', '')] == 'proton'
    df[('slc', 'reco', 'np', '', '', '')] = mask.groupby(level=[0,1,2]).transform('sum')

    mask = df[('pfp', 'trk', 'particle_reco', '', '', '')] == 'muon'
    df[('slc', 'reco', 'nmu', '', '', '')] = mask.groupby(level=[0,1,2]).transform('sum')

    mask = df[('pfp', 'trk', 'particle_reco', '', '', '')] == 'other'
    df[('slc', 'reco', 'n_oth', '', '', '')] = mask.groupby(level=[0,1,2]).transform('sum')

    return df


### Particle_classification() adds the reco particle columns. These selection codes can be applied after that
def muNp0pi_selection(df):
    muNp0pi = (df[('slc','reco','np')] >= 1) & (df[('slc','reco','nmu')] == 1) & (df[('slc','reco','n_oth')] == 0)
    muNp0pidf = df[muNp0pi]

    return muNp0pidf

def gump_selection(df):
    gump = (df[('slc','reco','np')] == 1) & (df[('slc','reco','nmu')] == 1) & (df[('slc','reco','n_oth')] == 0)
    gumpdf = df[gump]

    return gumpdf

def get_nuE_reco(df):

    PROTON_MASS = 0.938272
    NEUTRON_MASS = 0.939565
    MUON_MASS = 0.105658
    PION_MASS = 0.139570
    MASS_A = 22*NEUTRON_MASS + 18*PROTON_MASS - 0.34381
    BE = 0.0295
    MASS_Ap = MASS_A - NEUTRON_MASS + BE

    def mag2d(x,y):
        return np.sqrt(x**2 + y**2)

    def neutrino_energy(mu_p, mu_dir_x, mu_dir_y, mu_dir_z, p_p, p_E, p_dir_x, p_dir_y, p_dir_z):
        mu_E = mag2d(mu_p, MUON_MASS)
        #p_E = mag2d(p_p, PROTON_MASS)

        dpT = transverse_kinematics(mu_p, mu_dir_x, mu_dir_y, mu_dir_z, p_p, p_dir_x, p_dir_y, p_dir_z)['del_Tp']
        ET = np.sqrt(dpT**2 + MASS_Ap**2) - MASS_Ap
        
        return mu_E + (df.slc.reco.np.groupby(level=[0,1,2]).first() * (p_E - PROTON_MASS)) + ET + BE

    def transverse_kinematics(mu_p, mu_dir_x, mu_dir_y, mu_dir_z, p_p, p_dir_x, p_dir_y, p_dir_z):
        mu_E = mag2d(mu_p, MUON_MASS)
        p_E = mag2d(p_p, PROTON_MASS)

        mu_p_x = mu_p * mu_dir_x
        mu_p_y = mu_p * mu_dir_y
        mu_p_z = mu_p * mu_dir_z
        mu_phi_x = mu_p_x/mag2d(mu_p_x, mu_p_y)
        mu_phi_y = mu_p_y/mag2d(mu_p_x, mu_p_y)

        p_p_x = p_p * p_dir_x
        p_p_y = p_p * p_dir_y
        p_p_z = p_p * p_dir_z
        p_phi_x = p_p_x/mag2d(p_p_x, p_p_y)
        p_phi_y = p_p_y/mag2d(p_p_x, p_p_y)

        mu_Tp_x = mu_phi_y*mu_p_x - mu_phi_x*mu_p_y
        mu_Tp_y = mu_phi_x*mu_p_x - mu_phi_y*mu_p_y
        mu_Tp = mag2d(mu_Tp_x, mu_Tp_y)


        p_Tp_x = p_phi_y*p_p_x - p_phi_x*p_p_y
        p_Tp_y = p_phi_x*p_p_x - p_phi_y*p_p_y
        p_Tp = mag2d(p_Tp_x, p_Tp_y)



        del_Tp_x = mu_Tp_x + p_Tp_x
        del_Tp_y = mu_Tp_y + p_Tp_y
        del_Tp = mag2d(del_Tp_x, del_Tp_y)


        del_alpha = np.arccos(-(mu_Tp_x*del_Tp_x + mu_Tp_y*del_Tp_y)/(mu_Tp*del_Tp))
        del_phi = np.arccos(-(mu_Tp_x*p_Tp_x + mu_Tp_y*p_Tp_y)/(mu_Tp*p_Tp))

        mu_E = mag2d(mu_p, MUON_MASS)

        p_E = mag2d(p_p, PROTON_MASS)

        R = MASS_A + mu_p_z + p_p_z - mu_E - p_E
        del_Lp = 0.5*R - mag2d(MASS_Ap, del_Tp)**2/(2*R)

        del_p = mag2d(del_Tp, del_Lp)


        return pd.Series({'del_p' : del_p, 
                        'del_Tp' : del_Tp, 
                        'del_phi' : del_phi, 
                        'del_alpha' : del_alpha, 
                        'mu_E' : mu_E, 
                        'p_E' : p_E})

    df = df.sort_index(level=[0,1,2])

    dir_x = df.pfp.trk.dir.x
    dir_y = df.pfp.trk.dir.y
    dir_z = df.pfp.trk.dir.z

    mu_dir_x = dir_x[df.pfp.trk.particle_reco =='muon'].to_numpy()
    mu_dir_y = dir_y[df.pfp.trk.particle_reco =='muon'].to_numpy()
    mu_dir_z = dir_z[df.pfp.trk.particle_reco =='muon'].to_numpy()

    p_dir_x = dir_x[df.pfp.trk.particle_reco =='proton'].groupby(level=[0,1,2]).mean().to_numpy()
    p_dir_y = dir_y[df.pfp.trk.particle_reco =='proton'].groupby(level=[0,1,2]).mean().to_numpy()
    p_dir_z = dir_z[df.pfp.trk.particle_reco =='proton'].groupby(level=[0,1,2]).mean().to_numpy()

    #display(len(p_dir_x),len(mu_dir_x), len(muNp0pidf.groupby(level=[0,1,2])))

    mu_p = df.pfp.trk.rangeP.p_muon[df.pfp.trk.particle_reco =='muon'].groupby(level=[0,1,2]).mean().to_numpy()
    p_p_series = df.pfp.trk.rangeP.p_proton[df.pfp.trk.particle_reco == 'proton']
    p_p = p_p_series.groupby(level=[0,1,2]).mean().to_numpy()

    p_E = (np.sqrt(p_p_series**2 + PROTON_MASS**2).groupby(level=[0,1,2]).mean().to_numpy())

    nuE_reco = neutrino_energy(mu_p, mu_dir_x, mu_dir_y, mu_dir_z, p_p, p_E, p_dir_x, p_dir_y, p_dir_z)

    df[('slc', 'reco', 'E', '', '', '')] = nuE_reco

    return df

def flatten_cols(df):
        index = df.index
        df = df.reset_index()
        # Join non-empty tuple parts with '_', strip leading/trailing underscores
        df.columns = [
            '_'.join(str(p) for p in col if str(p).strip()).strip('_')
            if isinstance(col, tuple) else str(col)
            for col in df.columns.to_flat_index()
        ]
        # Deduplicate column names if any clash after flattening
        seen = {}
        new_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
        df.index=index
        return df
