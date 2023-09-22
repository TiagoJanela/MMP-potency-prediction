#
#  Copyright (C) 2019 Martin Vogt, University of Bonn
#
#  The contents are covered by the terms of the MIT License
#  See: https://opensource.org/licenses/MIT
#

import os
import re
import pickle
from multiprocessing import Pool
from operator import itemgetter, attrgetter
from functools import reduce
from itertools import combinations
from collections import Counter
import time
from sys import stderr
# import warnings

import logging

from typing import Tuple, Dict, List, Union, Optional, Set, Callable, cast

from collections import OrderedDict, defaultdict, Counter
# noinspection PyPackageRequirements
import rdkit.Chem as Chem
# noinspection PyPackageRequirements
from rdkit.Chem.rdchem import Mol, RWMol
from .fragmentation import fragment, get_acyclic_single_bonds, get_synthesizable_single_bonds, \
    count_substitution_sites, count_heavy_atoms_and_substitution_sites, CuttableBond, \
    count_heavy_atoms, count_heavy_atoms_and_negative_substitution_sites, \
    get_recap_single_bonds, reconstruct_molecule
from .mappedheap import MappedHeap

from .core_abstractions import CoreAbstraction
from .util import _objectview, NamedSmiles, NamedSmilesWithCore, FragmentationWithKey, Fragmentation, _true_fct, \
    head, hydrogenize_core, protonate_atom, get_attachment_points_and_remove_r_atoms, \
    add_r_atoms, get_rsites, to_rsite_map, to_generic_rsite_map, generic_rsite_map_to_fragment_smiles, swap_rsites, \
    renumber_r_atoms

debug_mode = False

logger = logging.getLogger(__name__)

periodic_table = Chem.GetPeriodicTable()


# logging.captureWarnings(True)


class CcrDictClass(dict):

    # defaultdict(list) functionality
    def __missing__(self, key):
        self[key] = list()
        return self[key]

    def get_fragmented_compound_list(self) -> List[FragmentationWithKey]:
        return [FragmentationWithKey(core, entry.smiles, entry.name, getattr(entry, 'core', core)) for core, frag_list
                in self.items() for entry in sorted(frag_list, key=lambda x: getattr(x, 'core', core))]

    def get_compound_to_core_map(self, unique_core=False) -> Dict[str, List[str]]:
        """
        From a CcrDict object relating cores to lists of compounds extract
        a compound map that relates compound names  to core Smiles
        :param self:
        :param unique_core:
        :return:
        """
        collection = self.items()
        if unique_core:
            cpd2core_map = dict()
            for core, substituents in collection:
                for s in substituents:
                    if s.name in cpd2core_map:
                        logger.warning("Compound '{}' is assigned to multiple cores".format(s.name))
                    rcore = getattr(s, 'core', core)
                    cpd2core_map[s.name] = core if rcore == core else (core, rcore)

        else:
            cpd2core_map = defaultdict(list)
            for core, substituents in collection:
                for s in substituents:
                    rcore = getattr(s, 'core', core)
                    cpd2core_map[s.name].append(core if rcore == core else (core, rcore))
        return cpd2core_map

    def to_csv(self, file, hydrogenized_core=True, name2smiles=None):
        if name2smiles:
            print(','.join(['core_key', 'name', 'smiles', 'core', 'substituents']), file=file)
        else:
            print(','.join(['core_key', 'name', 'core', 'substituents']), file=file)
        for core_key, substituents, name, core in self.get_fragmented_compound_list():
            if hydrogenized_core:
                core_key = hydrogenize_core(core_key)
            if name2smiles:
                smiles = name2smiles[name]
                print(core_key + "," + name + "," + smiles + "," + core + "," + substituents, file=file)
            else:
                print(core_key + "," + name + "," + core + "," + substituents, file=file)

    def to_dataframe(self):
        import pandas as pd
        from rdkit.Chem import PandasTools
        df = pd.DataFrame(self.get_fragmented_compound_list())
        PandasTools.AddMoleculeColumnToFrame(df, "core", "core_mol")
        PandasTools.AddMoleculeColumnToFrame(df, "substituents", "subst_mol")
        return df

    def to_rgroup_table(self, core):
        import pandas as pd
        from rdkit.Chem import PandasTools
        sub_patt = re.compile(r'^(.*)\[\*:(\d+)](.*)$')
        if core not in self:
            return None
        values = self[core]
        num_sites = core.count('*')
        d = {'core': [], 'name': []}
        for i in range(1, num_sites + 1):
            d[f"R{i}"] = []
        for frag in self[core]:
            if isinstance(frag, NamedSmilesWithCore):
                d['core'].append(frag.core)
            else:
                d['core'].append(core)
            d['name'].append(frag.name)
            subs = frag.smiles.split('.')
            assert (len(subs) == num_sites)
            for s in subs:
                m = sub_patt.match(s)
                key = f"R{m.group(2)}"
                if m.group(1) == m.group(3) == "":
                    d[key].append("[H]" + s)
                else:
                    d[key].append(s)
        l = len(d['core'])
        assert (all(len(v) == l for v in d.values()))
        df = pd.DataFrame(d)
        PandasTools.AddMoleculeColumnToFrame(df, "core", "core_mol")
        mol_cols = ["name", "core_mol"]
        cols = ["name", "core"]
        for i in range(1, num_sites + 1):
            r = f"R{i}"
            r_mol = r + "_mol"
            PandasTools.AddMoleculeColumnToFrame(df, r, r_mol)
            mol_cols.append(r_mol)
            cols.append(r)
        return df, mol_cols, cols


CcrDict = CcrDictClass[str, List[Fragmentation]]

CcrList = List[Tuple[str, List[Fragmentation]]]


def get_fragmented_compound_list(series: CcrDict) -> List[FragmentationWithKey]:
    return [FragmentationWithKey(core, entry.smiles, entry.name, getattr(entry, 'core', core)) for core, frag_list
            in series.items() for entry in sorted(frag_list, key=lambda x: getattr(x, 'core', core))]


def get_compound_to_core_map(series: CcrDict, unique_core=False) -> Dict[str, List[str]]:
    """
    From a CcrDict object relating cores to lists of compounds extract
    a compound map that relates compound names  to core Smiles
    :param series:
    :param unique_core:
    :return:
    """
    collection = series.items() if isinstance(series, dict) else series
    if unique_core:
        cpd2core_map = dict()
        for core, substituents in collection:
            for s in substituents:
                if s.name in cpd2core_map:
                    logger.warning("Compound '{}' is assigned to multiple cores".format(s.name))
                rcore = getattr(s, 'core', core)
                cpd2core_map[s.name] = core if rcore == core else (core, rcore)

    else:
        cpd2core_map = defaultdict(list)
        for core, substituents in collection:
            for s in substituents:
                rcore = getattr(s, 'core', core)
                cpd2core_map[s.name].append(core if rcore == core else (core, rcore))
    return cpd2core_map


def replace_r_site(core: str, idx: int, frag: str) -> str:
    mol = RWMol(Chem.MolFromSmiles(core + "." + frag))
    r_atoms = cast("Tuple[Atom,Atom]", tuple(
        a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == idx))
    assert len(r_atoms) == 2, "S:" + core + "." + frag + " I:" + str(idx) + " " + Chem.MolToSmiles(
        mol)
    if r_atoms[0].GetDegree() == 0 or r_atoms[1].GetDegree() == 0:  # Hydrogen cut
        atom = r_atoms[0].GetNeighbors()[0] if r_atoms[0].GetDegree() != 0 else r_atoms[1].GetNeighbors()[0]
        mol.RemoveAtom(r_atoms[1].GetIdx())
        mol.RemoveAtom(r_atoms[0].GetIdx())
        protonate_atom(atom)
    else:
        cut = CuttableBond(mol, None, idx, r_atoms)
        cut.undo_cut()
    return Chem.MolToSmiles(mol)


def remove_redundant_r_sites(core: str, series: List[Fragmentation]):
    """
    Identify substitution sites that show now variation across the series and eliminate them
    by extending the core accordingly
    :param core:
    :param series:
    :return:
    """
    oldcore = core
    rcores = {entry.core: entry.core for entry in series if hasattr(entry, 'core')}
    # rcores.pop(core,None) # Sadly cannot use rcores.discard(core) # Just avoids a little redundant work
    frag_sets = {r: set() for r in get_rsites(core)}
    for fragments in series:
        sc = to_rsite_map(fragments.smiles)
        for idx, frag in sc.items():
            frag_sets[idx].add(frag)
    remap = {}
    ct = 0
    pruned = False
    for idx in sorted(frag_sets):
        if len(frag_sets[idx]) == 1:
            logger.debug(f"for core {core}: redundant site {idx} detected")
            frag = head(frag_sets[idx])
            core = replace_r_site(core, idx, frag)
            rcores = {k: replace_r_site(c, idx, frag) for k, c in rcores.items()}
            pruned = True
        else:
            ct += 1
            remap[idx] = ct
    if pruned:
        # rcores[oldcore] = core # Just avoid a little redundant work
        core = renumber_r_atoms(core, remap)
        rcores = {k: renumber_r_atoms(c, remap) for k, c in rcores.items()}
        new_series = []
        for entry in series:
            fragments = entry.smiles
            name = entry.name
            sc = to_rsite_map(fragments)
            fragments = ".".join(frag for idx, frag in sc.items() if idx in remap)
            if hasattr(entry, 'core'):
                new_series.append(NamedSmilesWithCore(renumber_r_atoms(fragments, remap), name, rcores[entry.core]))
            else:
                new_series.append(NamedSmiles(renumber_r_atoms(fragments, remap), name))
        logger.debug(oldcore + " -> " + core)
        logger.debug(str(list(zip(series, new_series))[:2]))
        return core, new_series
    else:
        return core, series


### Main workflow functions

def read_smiles(supplier: Chem.SmilesMolSupplier, remove_stereochemistry=True,
                mol_filter: Callable[[Mol], bool] = _true_fct) -> Tuple[
    List[NamedSmiles], Dict[str, List[str]], List[str]]:
    """
    read in smiles
    remove stereochemical information/hydrogens
    (deuteriums should not be removed)
    :param remove_stereochemistry:
    :param mol_filter:
    :param supplier:
    :return:
    """
    all_smiles = OrderedDict()
    duplicates = defaultdict(list)
    failed_mols = []
    mol: Mol
    for mol in supplier:
        if not mol:
            print("Bad smiles found")
            continue
        name = mol.GetProp('_Name') if mol.HasProp("_Name") else Chem.MolToSmiles(mol)
        if mol_filter(mol):
            m = Chem.RemoveHs(mol)
            if remove_stereochemistry:
                Chem.RemoveStereochemistry(m)
            smiles = Chem.MolToSmiles(m)
            if '.' not in smiles:
                if smiles in all_smiles:
                    key_name = all_smiles[smiles]
                    duplicates[key_name].append(name)
                else:
                    all_smiles[smiles] = name
            else:
                failed_mols.append(name)
        else:
            failed_mols.append(name)
    return list(NamedSmiles(smiles, name) for smiles, name in all_smiles.items()), duplicates, failed_mols


def fragment_smiles(smiles_list: List[NamedSmiles], mode: str, cut_type: str, max_cuts: int,
                    min_rel_core_size: float, max_frag_size: int, max_time: int) -> CcrDict:
    mode = mode.upper()
    cut_type = cut_type.upper()
    assert mode.upper() in ["CCR", "MMP"]
    assert cut_type in ["SINGLE", "RECAP", "SYNTHESIZABLE"]
    get_cuttable_bonds = (get_acyclic_single_bonds if cut_type == "SINGLE" else
                          (get_synthesizable_single_bonds if cut_type == "SYNTHESIZABLE" else get_recap_single_bonds))
    all_cuts = CcrDict()
    for smi, ident in smiles_list:
        frags = fragment(smi, max_cuts, mode, get_cuttable_bonds, min_rel_core_size, max_frag_size, max_time)
        if isinstance(frags, dict):
            for core, substituents in frags.items():
                all_cuts[core].append(NamedSmiles(substituents, ident))
            # print(f"identified {len(frags)} cuts for {ident} ({smi})")
        else:
            print(f"timeout ({max_time} s) for {ident} after generating {frags} cuts ({smi})")
    return all_cuts


def fragment_for_mp(pars: Tuple):
    return pars[0], fragment(*pars[1:])


def fragment_parameter_wrapper(smiles_list: List[NamedSmiles], *pars):
    for smiles, ident in smiles_list:
        yield (ident, smiles) + pars


def fragment_smiles_mp(smiles_list: List[NamedSmiles], mode: str, cut_type: str, max_cuts: int,
                       min_rel_core_size: float, max_frag_size: int, max_time: int) -> CcrDict:
    mode = mode.upper()
    cut_type = cut_type.upper()
    assert mode.upper() in ["CCR", "MMP"]
    assert cut_type in ["SINGLE", "RECAP", "SYNTHESIZABLE"]
    get_cuttable_bonds = (get_acyclic_single_bonds if cut_type == "SINGLE" else
                          (get_synthesizable_single_bonds if cut_type == "SYNTHESIZABLE" else get_recap_single_bonds))
    all_cuts = CcrDict()
    parameter_generator = fragment_parameter_wrapper(smiles_list, max_cuts, mode, get_cuttable_bonds, min_rel_core_size,
                                                     max_frag_size, max_time)
    with Pool() as p:
        result = p.imap_unordered(fragment_for_mp, parameter_generator)
        for ident, frags in result:
            if isinstance(frags, dict):
                for core, substituents in frags.items():
                    all_cuts[core].append(NamedSmiles(substituents, ident))
                # print(f"identified {len(frags)} cuts for {ident} ({smi})")
            else:
                print(f"timeout ({max_time} s) for {ident} after generating {frags} cuts")
    return all_cuts


def generate_hcores(all_cuts: CcrDict, smiles_list: List[NamedSmiles],
                    virtualize_core: Optional[CoreAbstraction] = None) -> Tuple[
    Dict[str, List[str]], CcrDict]:
    if virtualize_core is None:
        hcores = defaultdict(list)
        for core in all_cuts:
            hcore = hydrogenize_core(core)
            hcores[hcore].append(core)
        for smi, ident in smiles_list:
            hcores[smi].append(smi)
            all_cuts[smi].append(NamedSmiles("", ident))
        return hcores, all_cuts
    else:
        new_cuts = CcrDict()
        hcores = defaultdict(set)
        for core, frags in all_cuts.items():
            _, vcore = virtualize_core(core)
            vcore = Chem.MolToSmiles(vcore)
            new_cuts[vcore].extend([NamedSmilesWithCore(frag.smiles, frag.name, core) for frag in frags])
            hcore = hydrogenize_core(vcore)
            hcores[hcore].add(vcore)
        for smi, ident in smiles_list:
            _, vcore = virtualize_core(smi)
            vcore = Chem.MolToSmiles(vcore)
            hcores[vcore].add(vcore)
            new_cuts[vcore].append(NamedSmilesWithCore("", ident, smi))
        hcores = {key: list(val) for key, val in hcores.items()}
        return hcores, new_cuts


def merge_single_hcore(cores: List[str], all_cuts: CcrDict,
                       virtualize_core: Optional[CoreAbstraction] = None) -> Optional[
    Tuple[str, List[NamedSmiles]]]:
    if not cores:
        return None
    cores = sorted(cores, key=count_substitution_sites)
    cms = list(map(get_attachment_points_and_remove_r_atoms, cores))
    ref_mol = cms[0][1]
    indices = list(map(lambda x: x.GetIdx(), ref_mol.GetAtoms()))
    renumberings = [dict() for _ in cms]
    new_r_idx = 1
    merged_sub_sites = dict()
    for idx in indices:
        index_increase = 0
        for i, (core, mol, ss) in enumerate(cms):
            if idx in ss:
                ci = new_r_idx
                index_increase = max(index_increase, len(ss[idx]))
                for oidx in ss[idx]:
                    renumberings[i][oidx] = ci
                    ci += 1
        if index_increase > 0:
            merged_sub_sites[idx] = list(range(new_r_idx, new_r_idx + index_increase))
            new_r_idx += index_increase
    ref_order = Chem.CanonicalRankAtoms(ref_mol)
    merged = add_r_atoms(ref_mol, merged_sub_sites)
    merged_core = Chem.MolToSmiles(merged)
    merged_substituents = {}
    merged_cores = {}
    for remap, core in zip(renumberings, cores):
        real_cores = {entry.core for entry in all_cuts[core] if hasattr(entry, 'core')}
        real_cores.discard(core)
        rc_map = {core: merged_core}
        for c0 in real_cores:
            _, t, _ = get_attachment_points_and_remove_r_atoms(c0)
            c = Chem.MolToSmiles(t)
            rc, vc = virtualize_core(c)
            vorder = Chem.CanonicalRankAtoms(vc)
            inv_rk = {rk: idx for idx, rk in enumerate(vorder)}
            mss = {inv_rk[ref_order[k]]: v for k, v in merged_sub_sites.items()}
            rc_map[c0] = Chem.MolToSmiles(add_r_atoms(rc, mss))
        for entry in all_cuts[core]:
            substituents = entry.smiles
            ident = entry.name
            real_core = rc_map[entry.core] if hasattr(entry, 'core') else None
            if ident not in merged_substituents:
                non_h_substituents = renumber_r_atoms(substituents, remap)
                h_substituents = ".".join("[*:{}]".format(x) for x in range(1, new_r_idx) if
                                          x not in remap.values())
                dot = "." if non_h_substituents and h_substituents else ""
                merged_substituents[ident] = non_h_substituents + dot + h_substituents
                if real_core is not None:
                    merged_cores[ident] = real_core
    return merged_core, list((NamedSmilesWithCore(smiles, ident,
                                                  merged_cores[ident]) if ident in merged_cores else NamedSmiles(smiles,
                                                                                                                 ident))
                             for ident, smiles in merged_substituents.items())


def merge_hcores(hcores: Dict[str, List[str]], all_cuts: CcrDict,
                 virtualize_core: Optional[CoreAbstraction] = None) -> CcrList:
    merged = []
    for hcore, cores in hcores.items():
        if len(cores) > 1:
            # print(f"merging:{hcore}, {len(cores)} cores, {cores}")
            merged.append(merge_single_hcore(cores, all_cuts, virtualize_core))
        else:
            merged.append((cores[0], all_cuts[cores[0]]))
    return merged


def remove_singletons(series: Union[CcrDict, CcrList]) -> CcrDict:
    collection = series.items() if isinstance(series, dict) else series
    return CcrDict(filter(lambda x: len(x[1]) > 1, collection))
    # return {core: compounds for core, compounds in collection if len(compounds) > 1}


def remove_sub_series(ccrs: CcrDict) -> CcrDict:
    cpd_to_cores = defaultdict(set)
    for core, substituents in ccrs.items():
        for s in substituents:
            cpd_to_cores[s.name].add(core)
    retained_cores = set()
    handled_cores = set()
    for core, substituents in ccrs.items():
        if core in handled_cores:
            continue
        common_cores = reduce(lambda x, y: x.intersection(y), map(lambda x: cpd_to_cores[x.name], substituents))
        if len(common_cores) == 1:
            retained_cores.add(head(common_cores))
        elif len(set(map(lambda x: len(ccrs[x]), common_cores))) == 1:  # Multiple series with same cpd set
            selected_core = max(common_cores, key=count_heavy_atoms_and_negative_substitution_sites)
            retained_cores.add(selected_core)
            handled_cores.update(common_cores)
    return CcrDict(filter(lambda x: x[0] in retained_cores, ccrs.items()))


def assign_compds_to_cores(ccrs: CcrDict) -> CcrDict:
    core_to_cpds = dict()
    for core, substituents in ccrs.items():
        core_to_cpds[core] = set(map(attrgetter('name'), substituents))
    cpd_to_cores = defaultdict(set)
    for core, substituents in ccrs.items():
        for s in substituents:
            cpd_to_cores[s.name].add(core)
    cores = list(ccrs.keys())
    heap = MappedHeap(cores, len(cores),
                      lambda core: (len(core_to_cpds[core]), count_heavy_atoms_and_substitution_sites(core), core))
    while heap.heap_size > 0:
        best_core = heap.extract_max()
        # print(best_core,len(core_to_cpds[best_core]))
        for cpd in core_to_cpds[best_core]:
            for core in cpd_to_cores[cpd]:
                if core == best_core:
                    continue
                if core in core_to_cpds:
                    core_to_cpds[core].remove(cpd)
                    if len(core_to_cpds[core]) <= 1:
                        heap.extract_key(core)
                        del core_to_cpds[core]
                    else:
                        heap.propagate_key(core)
    unique_ccrs = CcrDict()
    for core, cpds in core_to_cpds.items():
        new_subs = filter(lambda x: x.name in cpds, ccrs[core])
        unique_ccrs[core] = list(new_subs)
    return unique_ccrs


def sanitize(core: str, series: List[Fragmentation]) -> Tuple[str, List[NamedSmiles]]:
    pre = post = ""  # debugging variables
    rsites_eliminated = 0
    site_swaps_performed = 0
    swaps_performed = 0

    core, series = remove_redundant_r_sites(core, series)
    r_count = count_substitution_sites(core)
    if r_count <= 1:
        return core, series
    # reconstructed = {entry.name: Chem.MolToSmiles(reconstruct_molecule(getattr(entry, 'core', core), entry.smiles)) for
    #                 entry in series}
    # Explicit loop for debugging
    reconstructed = {}
    for entry in series:
        c = getattr(entry, 'core', core)
        rm = reconstruct_molecule(getattr(entry, 'core', core), entry.smiles)
        if rm is None:
            logger.error(f"core: {core} c: {c} s: {entry.smiles} n: {entry.name}")
            raise RuntimeError("Corrupted Smiles detected")
        reconstructed[entry.name] = Chem.MolToSmiles(rm)

    name2cores = {entry.name: entry.core for entry in series if hasattr(entry, 'core')}
    rsites_to_check = set(range(1, r_count + 1))
    fragments = {entry.name: to_generic_rsite_map(entry.smiles) for entry in series}

    while len(rsites_to_check) > 1:
        r_site_variations: Dict[int, Dict[str, Set[int]]] = dict()
        # Comment: dict[r_site][name] = set(swappable r-sites)
        for rsite_a in rsites_to_check:
            d: Dict[str, Set[int]] = dict()
            for entry in series:
                frags = entry.smiles
                name = entry.name
                rcore = getattr(entry, 'core', core)
                swappable: Set[int] = set()
                for rsite_b in rsites_to_check:
                    if (rsite_b == rsite_a or (
                            fragments[name][rsite_a] != fragments[name][rsite_b] and
                            Chem.MolToSmiles(
                                reconstruct_molecule(swap_rsites(rcore, rsite_a, rsite_b), fragments[name])) ==
                            reconstructed[name]
                    )):
                        swappable.add(rsite_b)
                d[name] = swappable
            r_site_variations[rsite_a] = d

        best_rsite = None  # head(rsites_to_check)
        best_consistency_count = 0
        best_fragment = ""
        change_needed = False
        for rsite in rsites_to_check:
            current_sites = Counter(fragments[name][rsite] for name in r_site_variations[rsite])

            pos_rsites = Counter()
            # pos_rsites = Counter(fragments[name][x] for name, rsite_set in r_site_variations[rsite].items()
            # for x in rsite_set if x in rsites_to_check)
            for name, rsite_set in r_site_variations[rsite].items():
                pos_rsites.update(set(fragments[name][x] for x in rsite_set if x in rsites_to_check))
            if len(pos_rsites) <= 1:
                best_rsite = rsite
                rsites_eliminated += 1
                if site_swaps_performed == 0:
                    logger.warning("Warning: Failed to remove redundant r-site in previous step")
                break
            else:
                dominating_fragment = pos_rsites.most_common(1)[0]
                current_most_frequent = current_sites.most_common(1)[0]
                if dominating_fragment[1] > best_consistency_count:
                    change_needed = dominating_fragment[1] > current_most_frequent[1]
                    best_fragment = dominating_fragment[0]
                    best_consistency_count = dominating_fragment[1]
                    best_rsite = rsite

        if change_needed:
            if best_consistency_count == len(fragments):
                rsites_eliminated += 1
            new_swaps_performed = 0
            for name, rset in r_site_variations[best_rsite].items():
                rset.intersection_update(rsites_to_check)
                to_swap = [x for x in rset if x != best_rsite and fragments[name][x] == best_fragment]
                if len(to_swap) and fragments[name][best_rsite] != fragments[name][to_swap[0]]:
                    to_swap = to_swap[0]
                    new_swaps_performed += 1
                    if debug_mode:
                        pre = Chem.MolToSmiles(reconstruct_molecule(core, fragments[name]))
                    fragments[name][to_swap], fragments[name][best_rsite] = \
                        fragments[name][best_rsite], fragments[name][to_swap]
                    if debug_mode:
                        post = Chem.MolToSmiles(reconstruct_molecule(core, fragments[name]))
                        if pre != post:
                            logger.warning(f"Inconsistency: pre: {pre} post: {post} {str(fragments[name])}")
                            logger.warning(f"n: {name} c: {core} site: {best_rsite} swap: {to_swap}")
                            logger.warning("core will not be optimized")
                            return core, series
            swaps_performed += new_swaps_performed
            if new_swaps_performed:
                site_swaps_performed += 1
        rsites_to_check.remove(best_rsite)

    if swaps_performed:
        logger.debug(f"R-sites optimized by performing {swaps_performed} swaps on {site_swaps_performed} sites")
        if rsites_eliminated:
            logger.debug(f"{rsites_eliminated} rsites have been eliminated")
        logger.debug(f"old core: {core}")
        logger.debug(f"old compounds {series}")
        series = [(NamedSmilesWithCore(generic_rsite_map_to_fragment_smiles(frags),
                                       name, name2cores[name]) if name in name2cores else NamedSmiles(
            generic_rsite_map_to_fragment_smiles(frags),
            name)) for name, frags in fragments.items()]
        new_core, new_series = remove_redundant_r_sites(core, series)
        if swaps_performed > 0:
            logger.debug(f"new core: {new_core}")
            logger.debug(f"new compounds {new_series}")
        return new_core, new_series
    else:
        return core, series


def sanitize_series(ccrs: CcrDict) -> CcrDict:
    # ccrs = dict(remove_redundant_r_sites(core, series) for core, series in ccrs.items())
    sanitized = [sanitize(core, series) for core, series in ccrs.items()]

    core_counts = Counter(item[0] for item in sanitized)
    duplicate_cores = [(core, count) for core, count in core_counts.items() if count > 1]
    if duplicate_cores:
        logger.debug("Duplicate cores detected: ", duplicate_cores)

    ccrs = CcrDict()
    for core, series in sanitized:
        if core in ccrs:
            core_series = ccrs[core]
            series_names = set(x.name for x in core_series)
            for compound in series:
                if compound.name not in series_names:
                    core_series.append(compound)
        else:
            ccrs[core] = series

    hcore_counts = Counter(hydrogenize_core(item[0]) for item in ccrs.items())
    duplicate_hcores = [(core, count) for core, count in hcore_counts.items() if count > 1]
    if duplicate_hcores:
        logger.debug("Duplicate hcores detected, merging required: ", duplicate_hcores)
        hcores, all_cuts = generate_hcores(ccrs, [])
        ccrs = CcrDict(merge_hcores(hcores, all_cuts))

    # ccrs = CcrDict(sanitize(core, series) for core, series in ccrs.items())
    return ccrs


def sanitize_series_using_sub_series(ccrs: CcrDict, sub_ccrs: CcrDict) -> CcrDict:
    chaanss = count_heavy_atoms_and_negative_substitution_sites
    cpd_to_cores = defaultdict(set)
    for core, substituents in sub_ccrs.items():
        for s in substituents:
            cpd_to_cores[s.name].add(core)
    opt_series: CcrDict = CcrDict()
    handled_cores = set()
    for core, substituents in ccrs.items():
        names = set(x.name for x in substituents)
        if core in handled_cores:
            continue
        common_cores = reduce(lambda x, y: x.intersection(y), map(lambda x: cpd_to_cores[x.name], substituents))
        selected_core = max(common_cores, key=chaanss)
        new_val = chaanss(selected_core)
        old_val = chaanss(core)
        if new_val > old_val:
            logger.debug(f"Replacing core: '{core}' with '{selected_core}")
            logger.debug(f"(heavy atoms,sub sites): {old_val} ->  {new_val}")
            opt_series[selected_core] = [x for x in sub_ccrs[selected_core] if x.name in names]
        else:
            opt_series[core] = substituents
    return sanitize_series(opt_series)


# Mutable default argument used here on purpose in order to
# remember basename on future calls
# noinspection PyDefaultArgument
def buffered(fct, modifier: str, basename: Optional[str] = None, static=[None]):
    if basename:
        # noinspection PyTypeChecker
        static[0] = basename
    else:
        basename = static[0]
    if not basename:
        return fct()
    filename = basename + "." + modifier + ".pickle"
    if os.path.exists(filename):
        return pickle.load(open(filename, "rb"))
    else:
        result = fct()
        with open(filename, "wb") as outf:
            pickle.dump(result, outf)
        return result


def verify_fragmentation(core: str, sub: Fragmentation, smiles_dict: Dict[str, str]) -> bool:
    core = getattr(sub, 'core', core)
    m = reconstruct_molecule(core, sub.smiles)
    if m:
        smiles = Chem.MolToSmiles(m)
        if smiles != smiles_dict[sub.name]:
            logger.warning(f"False Reconstruction: Name: {sub.name} Expected: {smiles_dict[sub.name]} Got: {smiles}\n"
                           f"from {core} and {sub.smiles}")
            return False
        return True
    else:
        logger.warning(f"Molecules could not be reconstructed from {core} and {sub.smiles}")
        return False


def verify_mmps(mmps: Dict[Tuple[str, str], Tuple[str, Fragmentation, Fragmentation]],
                smiles_list: List[NamedSmiles]) -> bool:
    failed_mol = 0
    failed_smiles = 0
    smiles_dict = {name: smiles for smiles, name in smiles_list}
    for key, (core, sub1, sub2) in mmps.items():
        for s in (sub1, sub2):
            okay = verify_fragmentation(core, s, smiles_dict)
            if not okay:
                if okay is None:
                    failed_mol += 1
                else:
                    failed_smiles += 1
    return failed_mol + failed_smiles == 0


def verify_series(ccr_dict: CcrDict, smiles_list: Dict[str, str]) -> bool:
    failed_mol = 0
    failed_smiles = 0
    smiles_dict = {name: smiles for smiles, name in smiles_list}
    for core, substituents in ccr_dict.items():
        for s in substituents:
            okay = verify_fragmentation(core, s, smiles_dict)
            if not okay:
                if okay is None:
                    failed_mol += 1
                else:
                    failed_smiles += 1
    return failed_mol + failed_smiles == 0


def generate_ccrs(supplier: Chem.SmilesMolSupplier, mode: str, cut_type: str, max_cuts: int, min_rel_core_size: float,
                  max_frag_size: int, max_time: int, mol_filter: Callable[[Mol], bool] = _true_fct,
                  basename: Optional[str] = None) -> Union[_objectview, Tuple[CcrDict, CcrDict]]:
    warn_msg = "'generate_ccrs' is deprecated, use 'run_ccr' or 'run_mmp' instead."
    # warnings.warn(warn_msg,category=DeprecationWarning)
    # logger.warning(warn_msg)
    print("WARNING", warn_msg, file=stderr)
    if mode.upper() == "MMP":
        return run_mmp(supplier, cut_type, max_cuts, min_rel_core_size, max_frag_size, -1, max_time, mol_filter,
                       basename)

    smiles_list, duplicates, failed_mols = buffered(lambda: read_smiles(supplier, mol_filter=mol_filter), "read_smiles",
                                                    basename)
    print("# smiles:", len(smiles_list))
    print("# duplicates:", sum(map(len, duplicates.values())))
    print("# discarded molecules:", len(failed_mols))

    start = time.time()
    all_cuts = buffered(
        lambda: fragment_smiles_mp(smiles_list, mode, cut_type, max_cuts, min_rel_core_size, max_frag_size, max_time),
        "fragment_smiles")
    end = time.time()
    print("# frames:", len(all_cuts))
    print("# cuts: ", sum(map(len, all_cuts.values())))
    print("Time: ", end - start)
    print("FPS: ", sum(map(len, all_cuts.values())) / (end - start))

    hcores, all_cuts = buffered(lambda: generate_hcores(all_cuts, smiles_list), "generate_hcores")
    print("# frames:", len(all_cuts))
    print("# cuts: ", sum(map(len, all_cuts.values())))
    print("# hcores: ", len(hcores))
    print("# cores in hcores: ", sum(map(len, hcores.values())))

    ccrs = buffered(lambda: merge_hcores(hcores, all_cuts), "merge_hcores")
    print("#hcores: ", len(ccrs), "#cpds: ", sum(map(len, map(itemgetter(1), ccrs))), "#unique cpds:",
          len(set(cpd.name for _, series in ccrs for cpd in series)))
    ccrs = dict(filter(lambda x: len(x[1]) > 1, ccrs))
    subseries_ccr = ccrs
    print("Removed single-cpd series:")
    print("#hcores: ", len(ccrs), "#cpds: ", sum(map(len, map(itemgetter(1), ccrs))), "#unique cpds:",
          len(set(cpd.name for _, series in ccrs for cpd in series)))
    ccrs = buffered(lambda: remove_sub_series(ccrs), "remove_sub_series")
    print("Removed sub-series:")
    print("#hcores: ", len(ccrs), "#cpds: ", sum(map(len, ccrs.values())), "#unique cpds:",
          len(set(cpd.name for series in ccrs.values() for cpd in series)))
    ccrs_clean = buffered(lambda: sanitize_series(ccrs), "sanitized_ccrs")
    print("Optimized R-sites")
    print("#hcores: ", len(ccrs_clean), "#cpds: ", sum(map(len, ccrs_clean.values())), "#unique cpds:",
          len(set(cpd.name for series in ccrs_clean.values() for cpd in series)))

    uccrs = buffered(lambda: assign_compds_to_cores(ccrs), "assign_compds_to_cores")
    print("Unique assignments:")
    print("#hcores: ", len(uccrs), "#cpds: ", sum(map(len, uccrs.values())), "#unique cpds:",
          len(set(cpd.name for series in uccrs.values() for cpd in series)))

    uccrs_clean = buffered(lambda: sanitize_series_using_sub_series(uccrs, subseries_ccr), "sanitized_unique_ccrs")
    print("Optimized R-sites for unique ccrs")
    print("#hcores: ", len(uccrs_clean), "#cpds: ", sum(map(len, uccrs_clean.values())), "#unique cpds:",
          len(set(cpd.name for series in uccrs_clean.values() for cpd in series)))

    return ccrs_clean, uccrs_clean


run_mmp_return_vals = ('mmps', 'mms', 'cuts', 'smiles', 'duplicates', 'failed')


def run_mmp(supplier: Chem.SmilesMolSupplier, cut_type: str, max_cuts: int, min_rel_core_size: float,
            max_frag_size: int, max_xchg_difference, max_time: int, mol_filter: Callable[[Mol], bool] = _true_fct,
            basename: Optional[str] = None,
            return_vals=('mmps', 'mms', 'cuts', 'smiles', 'duplicates', 'failed')) -> _objectview:
    return_dict = {}
    smiles_list, duplicates, failed_mols = buffered(lambda: read_smiles(supplier, mol_filter=mol_filter),
                                                    "1_read_smiles",
                                                    basename)
    if 'smiles' in return_vals:
        return_dict['smiles'] = smiles_list
    if 'duplicates' in return_vals:
        return_dict['duplicates'] = duplicates
    if 'failed' in return_vals:
        return_dict['failed'] = failed_mols

    print("# smiles:", len(smiles_list))
    print("# duplicates:", sum(map(len, duplicates.values())))
    print("# discarded molecules:", len(failed_mols))

    start = time.time()
    all_cuts = buffered(
        lambda: fragment_smiles_mp(smiles_list, "MMP", cut_type, max_cuts, min_rel_core_size, max_frag_size, max_time),
        "2_fragment_smiles")
    end = time.time()

    smiles_dict = dict(smiles_list)
    for core, val in all_cuts.items():
        if count_substitution_sites(core) == 1:
            smiles_name = smiles_dict.get(hydrogenize_core(core), None)
            if smiles_name is not None:
                val.append(NamedSmiles('[*:1]', smiles_name))
    if 'cuts' in return_vals:
        return_dict['cuts'] = all_cuts
    print("Cuts")
    print("# frames:", len(all_cuts))
    print("# cuts: ", sum(map(len, all_cuts.values())))
    print("Time: ", end - start)
    print("FPS: ", sum(map(len, all_cuts.values())) / (end - start))

    all_mms: Dict[str, List[NamedSmiles]] = dict(filter(lambda x: len(x[1]) > 1, all_cuts.items()))
    print("Raw MMS")
    print("# frames:", len(all_mms))
    print("# cpds: ", sum(map(len, all_mms.values())))

    mmps = {}
    for core, frags in all_mms.items():
        for cpds in combinations(frags, 2):
            cpd1, cpd2 = sorted(cpds, key=lambda x: x.name)
            key = (cpd1.name, cpd2.name)
            if cpd1.name == cpd2.name:
                logger.warning("Compound is self-mmp:", core, cpd1, cpd2)
                continue
            hvy1, cut_ct = count_heavy_atoms_and_substitution_sites(cpd1.smiles)
            hvy2, cut_ct = count_heavy_atoms_and_substitution_sites(cpd2.smiles)
            if max_xchg_difference >= 0 and abs(hvy1 - hvy2) > max_xchg_difference:
                continue
            if key not in mmps:
                mmps[key] = (core, cpd1, cpd2)
            else:
                new_hvy = count_heavy_atoms(core)
                old_hvy = count_heavy_atoms(mmps[key][0])

                if new_hvy > old_hvy or new_hvy == old_hvy and count_substitution_sites(mmps[key][0]) > cut_ct:
                    mmps[key] = (core, cpd1, cpd2)
    if 'mmps' in return_vals:
        return_dict['mmps'] = mmps
    print("MMPs")
    print("# MMPs:", len(mmps))

    mms = remove_sub_series(all_mms)
    if 'mms' in return_vals:
        return_dict['mms'] = mms
    print("MMS")
    print("# frames:", len(mms))
    print("# cpds: ", sum(map(len, mms.values())))
    return _objectview(return_dict)


run_ccr_return_vals = (
    'unique', 'unique_raw', 'overlapping', 'overlapping_raw', 'sub_series', 'all_series', 'cuts', 'smiles',
    'duplicates', 'failed')


def run_ccr(supplier: Chem.SmilesMolSupplier, cut_type: str, max_cuts: int, min_rel_core_size: float,
            max_frag_size: int, max_time: int, mol_filter: Callable[[Mol], bool] = _true_fct,
            virtualize_core: Optional[CoreAbstraction] = None,
            basename: Optional[str] = None,
            return_vals=(
                    'unique', 'overlapping', 'sub_series', 'all_series', 'cuts', 'smiles', 'duplicates', 'failed')) -> \
        _objectview:
    return_dict = {}
    smiles_list, duplicates, failed_mols = buffered(lambda: read_smiles(supplier, mol_filter=mol_filter),
                                                    "1_read_smiles",
                                                    basename)
    if 'smiles' in return_vals:
        return_dict['smiles'] = smiles_list
    if 'duplicates' in return_vals:
        return_dict['duplicates'] = duplicates
    if 'failed' in return_vals:
        return_dict['failed'] = failed_mols

    print("# smiles:", len(smiles_list))
    print("# duplicates:", sum(map(len, duplicates.values())))
    print("# discarded molecules:", len(failed_mols))

    start = time.time()
    all_cuts = buffered(
        lambda: fragment_smiles_mp(smiles_list, "CCR", cut_type, max_cuts, min_rel_core_size, max_frag_size, max_time),
        "2_fragment_smiles")
    end = time.time()
    if 'cuts' in return_vals:
        return_dict['cuts'] = all_cuts
    print("# frames:", len(all_cuts))
    print("# cuts: ", sum(map(len, all_cuts.values())))
    print("Time: ", end - start)
    print("FPS: ", sum(map(len, all_cuts.values())) / (end - start))

    hcores, all_cuts = buffered(lambda: generate_hcores(all_cuts, smiles_list, virtualize_core), "3_generate_hcores")
    print("# frames:", len(all_cuts))
    print("# cuts: ", sum(map(len, all_cuts.values())))
    print("# hcores: ", len(hcores))
    print("# cores in hcores: ", sum(map(len, hcores.values())))

    ccr_list = buffered(lambda: merge_hcores(hcores, all_cuts, virtualize_core), "4_merge_hcores")
    if 'all_series' in return_vals:
        return_dict['all_series'] = CcrDict(ccr_list)
    print("#hcores: ", len(ccr_list), "#cpds: ", sum(map(len, map(itemgetter(1), ccr_list))), "#unique cpds:",
          len(set(cpd.name for _, series in ccr_list for cpd in series)))
    ccrs = CcrDict(filter(lambda x: len(x[1]) > 1, ccr_list))
    sub_series = ccrs
    if 'sub_series' in return_vals:
        return_dict['sub_series'] = ccrs  # or: sanitize_series(ccrs)
    print("Removed single-cpd series:")
    print("#hcores: ", len(ccrs), "#cpds: ", sum(map(len, ccrs.values())), "#unique cpds:",
          len(set(cpd.name for series in ccrs.values() for cpd in series)))

    ccrs = buffered(lambda: remove_sub_series(ccrs), "5_remove_sub_series")
    print("Removed sub-series:")
    print("#hcores: ", len(ccrs), "#cpds: ", sum(map(len, ccrs.values())), "#unique cpds:",
          len(set(cpd.name for series in ccrs.values() for cpd in series)))

    if 'overlapping_raw' in return_vals:
        return_dict['overlapping_raw'] = CcrDict(ccrs)
    ccrs_clean = buffered(lambda: sanitize_series(ccrs), "6_sanitized_ccrs")
    if 'overlapping' in return_vals:
        return_dict['overlapping'] = CcrDict(ccrs_clean)
    print("Optimized R-sites")
    print("#hcores: ", len(ccrs_clean), "#cpds: ", sum(map(len, ccrs_clean.values())), "#unique cpds:",
          len(set(cpd.name for series in ccrs_clean.values() for cpd in series)))

    uccrs = buffered(lambda: assign_compds_to_cores(ccrs), "7_assign_compds_to_cores")
    if 'unique_raw' in return_vals:
        return_dict['unique_raw'] = CcrDict(uccrs)
    print("Unique assignments:")
    print("#hcores: ", len(uccrs), "#cpds: ", sum(map(len, uccrs.values())), "#unique cpds:",
          len(set(cpd.name for series in uccrs.values() for cpd in series)))
    uccrs_clean = buffered(lambda: sanitize_series_using_sub_series(uccrs, sub_series),
                           "8_sanitized_unique_ccrs")
    # Sanitation w/o sub-series check:
    # uccrs_clean = buffered(lambda: sanitize_series(uccrs), "sanitized_unique_ccrs")
    if 'unique' in return_vals:
        return_dict['unique'] = CcrDict(uccrs_clean)
    print("Optimized R-sites for unique ccrs")
    print("#hcores: ", len(uccrs_clean), "#cpds: ", sum(map(len, uccrs_clean.values())), "#unique cpds:",
          len(set(cpd.name for series in uccrs_clean.values() for cpd in series)))

    return _objectview(return_dict)
