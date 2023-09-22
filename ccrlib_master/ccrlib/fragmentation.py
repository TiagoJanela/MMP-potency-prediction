#
#  Copyright (C) 2019 Martin Vogt, University of Bonn
#
#  The contents are covered by the terms of the MIT License
#  See: https://opensource.org/licenses/MIT
#
from collections import defaultdict
from typing import Tuple, Dict, List, NamedTuple, Optional, Union
from rdkit import Chem as Chem
from operator import itemgetter

from rdkit.Chem import ChiralType, Mol
from rdkit.Chem.rdchem import Atom, Bond, BondType, RWMol
import re
import time
import logging

from .util import renumber_r_atoms, protonate_atom, to_specific_rsite_map

from .reactions import IsSynthesizable

logger = logging.getLogger(__name__)

stereo_awareness = False
"""
Alternative approach to generate cores by RECAP fragmentation:

def recap_fragment(smi, min_core_size=2 / 3):
    mol: Mol = Chem.MolFromSmiles(smi)  # type: Mol
    # if mol is None: return None
    tot = mol.GetNumHeavyAtoms() * min_core_size
    return list(filter(lambda x: Chem.MolFromSmiles(x).GetNumHeavyAtoms() >= tot,
                       Chem.Recap.RecapDecompose(mol).GetAllChildren().keys()))
"""


def add_atom(mol: RWMol, atomic_num: int, map_idx: int) -> Atom:
    a = Atom(atomic_num)
    a.SetAtomMapNum(map_idx)
    idx = mol.AddAtom(a)
    return mol.GetAtomWithIdx(idx)


def add_r_atom_to_mol(mol: RWMol, map_idx: int) -> Atom:
    return add_atom(mol, 0, map_idx)


def add_bond(mol: RWMol, bond_order: BondType, a1: Atom, a2: Atom) -> int:
    return mol.AddBond(a1.GetIdx(), a2.GetIdx(), bond_order)


def remove_bond(mol: RWMol, a1: Atom, a2: Atom) -> int:
    return mol.RemoveBond(a1.GetIdx(), a2.GetIdx())


def get_bond(mol: RWMol, a1: Atom, a2: Atom) -> Bond:
    return mol.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx())


def get_bond_atoms(bond: Bond) -> Tuple[Atom, Atom]:
    begin: Atom = bond.GetBeginAtom()
    end: Atom = bond.GetEndAtom()
    if end.GetIdx() < begin.GetIdx():
        return end, begin
    else:
        return begin, end


class AtomChirality(NamedTuple):
    type: ChiralType = ChiralType.CHI_UNSPECIFIED
    neighbors: List[int] = []

    def __bool__(self):
        return self.type != ChiralType.CHI_UNSPECIFIED


def get_atom_chirality(atom: Atom) -> AtomChirality:
    chiral_tag = atom.GetChiralTag()
    if chiral_tag != ChiralType.CHI_UNSPECIFIED:
        return AtomChirality(chiral_tag, [n.GetIdx() for n in atom.GetNeighbors()])
    else:
        return AtomChirality()


# TODO: Handle hydrogen cuts ?
class CuttableBond:
    """
    Class representing bonds (represented by a pair of atoms)) of a molecule that can be removed (i.e., cut) and
    replaced by dummy atoms (i.e., attachment points)
    """

    chirality_map = {ChiralType.CHI_TETRAHEDRAL_CW: [ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW],
                     ChiralType.CHI_TETRAHEDRAL_CCW: [ChiralType.CHI_TETRAHEDRAL_CCW, ChiralType.CHI_TETRAHEDRAL_CW],
                     }

    def __init__(self, mol: RWMol, atoms: Optional[Tuple[Atom, Atom]], map_idx: int,
                 r_atoms: Optional[Tuple[Atom, Atom]] = None, stereo_aware=stereo_awareness):
        """

        :param mol: Molecule
        :param atoms: pair of atoms forming a bond
        :param map_idx: map index of the attachment point dummy atoms
        :param r_atoms: If None, the bond ist not cut, else the tuple is the pair of dummy atoms attached
                        to the atoms of the cut
        """
        self.mol: RWMol = mol
        self.atoms: Optional[Tuple[Atom, Atom]] = atoms
        self.map_idx: int = map_idx
        self.r_atoms: Optional[Tuple[Atom, Atom]] = r_atoms
        self.stereo_aware = stereo_aware

    def cut_bond(self):
        """
        Cut a bond.
        :return: None
        """
        assert self.r_atoms is None
        rp0 = add_r_atom_to_mol(self.mol, self.map_idx)
        rp1 = add_r_atom_to_mol(self.mol, self.map_idx)
        self.r_atoms = (rp0, rp1)
        bond_type = get_bond(self.mol, *self.atoms).GetBondType()
        if self.stereo_aware:
            chiralities = [get_atom_chirality(a) for a in self.atoms]
            remove_bond(self.mol, *self.atoms)
            add_bond(self.mol, bond_type, self.atoms[0], rp0)
            add_bond(self.mol, bond_type, self.atoms[1], rp1)
            for atom, other, chirality in zip(self.atoms, reversed(self.atoms), chiralities):
                if chirality:
                    idx = chirality.neighbors.index(other.GetIdx())
                    flag = (idx % 2 == len(chirality.neighbors) % 2)
                    atom.SetChiralTag(CuttableBond.chirality_map[chirality.type][flag])
        else:
            remove_bond(self.mol, *self.atoms)
            add_bond(self.mol, bond_type, self.atoms[0], rp0)
            add_bond(self.mol, bond_type, self.atoms[1], rp1)

    def undo_cut(self):
        """
        Rejoin two atoms of a bond
        :return: None
        """
        assert self.r_atoms is not None
        if self.atoms is None:
            self.atoms = (self.r_atoms[0].GetNeighbors()[0], self.r_atoms[1].GetNeighbors()[0])
        # print("removing",self.r_atoms)
        bond_type = get_bond(self.mol, self.atoms[0], self.r_atoms[0]).GetBondType()
        if self.stereo_aware:
            chiralities = [get_atom_chirality(a) for a in self.atoms]
            r_indices = [r.GetIdx() for r in self.r_atoms]
            self.mol.RemoveAtom(self.r_atoms[0].GetIdx())
            self.mol.RemoveAtom(self.r_atoms[1].GetIdx())
            add_bond(self.mol, bond_type, *self.atoms)
            for atom, other_idx, chirality in zip(self.atoms, r_indices, chiralities):
                if chirality:
                    idx = chirality.neighbors.index(other_idx)
                    flag = (idx % 2 == len(chirality.neighbors) % 2)
                    atom.SetChiralTag(CuttableBond.chirality_map[chirality.type][flag])
        else:
            self.mol.RemoveAtom(self.r_atoms[0].GetIdx())
            self.mol.RemoveAtom(self.r_atoms[1].GetIdx())
            add_bond(self.mol, bond_type, *self.atoms)
            self.r_atoms = None


def ranks_to_order(ranks):
    """
    ranks of atoms are convert to an ordering of the atoms according to these ranks, e.g.
    if the ranks are [3, 0, 2, 1] for atoms [0, 1, 2, 3], the ordering would be [1, 3, 2 ,0]
    Note, that Chem.CanonicalRankAtoms(m) results in atom ranks while
    Chem.RenumberAtoms(m,order) requires the atom order.
    Note, eval(mol.GetProp("_smilesAtomOutputOrder")) gives the order of atoms and can be used directly
    for renumbering.
    :param ranks: list of ranks for the atoms 0 to n-1
    :return: order of atoms

    >>> ranks_to_order([3, 0, 2, 1])
    [1, 3, 2, 0]
    >>> ranks_to_order([1, 3, 2 ,0])
    [3, 0, 2, 1]
    >>> ranks_to_order([3, 2, 1, 0])
    [3, 2, 1, 0]
    """
    order = ranks[:]  # create list of length len(ranks) (faster than [0]*len(ranks))
    for a, r in enumerate(ranks):
        order[r] = a
    return order


def count_heavy_atoms(smi: str):
    """
    Count the number of heavy atoms in a Smiles string

    :param smi: Smiles
    :return: number of heavy atoms

    >>> count_heavy_atoms("Oc1cc(N)ccc1C")
    9
    >>> count_heavy_atoms("[CH3]N1C(=NC(C1=O)(c2ccccc2)c3cc[cH]cc3)N")
    20
    >>> count_heavy_atoms("CC[C@H](C)[C@@H]1C(=O)N[C@@H](C(=O)N1[C@H](C2=COC(=N2)C)C(=O)N3CCOCC3)C4CC5=CC=CC=C5C4")
    36
    """
    # assumes only implicit hydrogens!
    in_bracket = 0
    heavy_count = 0
    for c in smi:
        if in_bracket:
            if in_bracket == 1 and c not in "HR*":
                heavy_count += 1
            in_bracket = 0 if c == "]" else in_bracket + 1
        elif c == "[":
            in_bracket = 1
        elif c.upper() in "BCNOSPFI":
            heavy_count += 1
    return heavy_count


def count_heavy_atoms_and_substitution_sites(smi: str):
    """
    Count the number of heavy atoms and attachment points in a Smiles string

    :param smi: Smiles
    :return: Tuple of heavy atom count and number of attachment points

    >>> count_heavy_atoms_and_substitution_sites("Oc1cc([*:2])ccc1[*:1]")
    (7, 2)
    >>> count_heavy_atoms_and_substitution_sites("[*:3]N1C(=NC(C1=O)(c2ccccc2)c3cc[cH]cc3)[*:1]")
    (18, 2)
    >>> count_heavy_atoms_and_substitution_sites("[*:1]C[C@H]([*:3])[C@@H]1C(=O)N[C@@H](C(=O)N1[C@H](C2=COC(=N2)[*:2])C(=O)N3CCOCC3)C4CC5=CC=CC=C5C4")
    (33, 3)
    """
    # assumes only implicit hydrogens!
    in_bracket = 0
    heavy_count = 0
    sub_sites = 0
    for c in smi:
        if in_bracket:
            if in_bracket == 1:
                if c not in "HR*":
                    heavy_count += 1
                elif c in "*R":
                    sub_sites += 1
            in_bracket = 0 if c == "]" else in_bracket + 1
        elif c == "[":
            in_bracket = 1
        elif c.upper() in "BCNOSPFI":
            heavy_count += 1
    return heavy_count, sub_sites


def count_heavy_atoms_and_negative_substitution_sites(smi: str):
    heavy_count, sub_sites = count_heavy_atoms_and_substitution_sites(smi)
    return heavy_count, -sub_sites


def count_substitution_sites(smi: str):
    """
    Count the number of attachment points.

    :param smi: Smiles
    :return: Number of substitution sites

    >>> count_substitution_sites("Oc1cc([*:2])ccc1[*:1]")
    2
    >>> count_substitution_sites("[*:3]N1C(=NC(C1=O)(c2ccccc2)c3cc[cH]cc3)[*:1]")
    2
    >>> count_substitution_sites("[*:1]C[C@H]([*:3])[C@@H]1C(=O)N[C@@H](C(=O)N1[C@H](C2=COC(=N2)[*:2])C(=O)N3CCOCC3)C4CC5=CC=CC=C5C4")
    3
    """
    return smi.count("[*:") + smi.count("[R")


def get_synthesizable_single_bonds(mol: RWMol) -> List[Tuple[Atom, Atom]]:
    """
    Determine all synthesizable acyclic single bonds of a molecule. The bonds must bot be contained in a ring and
    have to be single bonds. The bonds have to be syntesizable according to a predefined set of rules.
    :param mol: Molecule
    :return: list of bonds given as atom pairs
    """
    removable_bonds = set()
    bond: Bond
    for bond in mol.GetBonds():
        if bond.GetBondType() == BondType.SINGLE and not bond.IsInRing():
            atoms = get_bond_atoms(bond)
            is_synthesizable = IsSynthesizable(mol, mol, *atoms, swap=False,
                                               track_reactions=False, allow_multiaromatic_N_O_S=False)
            if is_synthesizable.detect():
                removable_bonds.add(atoms)
            else:
                is_synthesizable = IsSynthesizable(mol, mol, *atoms, swap=True,
                                                   track_reactions=False, allow_multiaromatic_N_O_S=False)
                if is_synthesizable.detect():
                    removable_bonds.add(atoms)

    return list(removable_bonds)


def get_recap_single_bonds(mol: RWMol) -> List[Tuple[Atom, Atom]]:
    """
    Determine all synthesizable acyclic single bonds of a molecule. The bonds must bot be contained in a ring and
    have to be single bonds. The bonds have to be syntesizable according to a predefined set of rules.
    :param mol: Molecule
    :return: list of bonds given as atom pairs
    """
    removable_bonds = set()
    bond: Bond
    for bond in mol.GetBonds():
        if bond.GetBondType() == BondType.SINGLE and not bond.IsInRing():
            atoms = get_bond_atoms(bond)
            is_synthesizable = IsSynthesizable(mol, mol, *atoms, swap=False,
                                               track_reactions=True, allow_multiaromatic_N_O_S=False)
            reaction = is_synthesizable.detect()
            if reaction is None:
                is_synthesizable = IsSynthesizable(mol, mol, *atoms, swap=True,
                                                   track_reactions=True, allow_multiaromatic_N_O_S=False)
                reaction = is_synthesizable.detect()
            if reaction and reaction[0] in ["suzuki", "disulfide", "aroN_aliC", "sulfonamide", "lactam", "amide",
                                            "thioamide", "ester",
                                            "thioester", "amine", "ether", "thioether"]:
                removable_bonds.add(atoms)

    return list(removable_bonds)


def get_acyclic_single_bonds(mol: RWMol) -> List[Tuple[Atom, Atom]]:
    """
    Determine all cuttable bonds of a molecule. The bonds must bot be contained in a ring and
    have to be single bonds. Bonds are returned a tuple of the atom start and end indices.
    :param mol: Molecule
    :return: list of bonds given as atom pairs
    """
    bond: Bond
    removable_bonds = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() == BondType.SINGLE and not bond.IsInRing():
            removable_bonds.add(get_bond_atoms(bond))
    return list(removable_bonds)


def canonicalize_ccr_cut(core: str, frags0: List[str], max_frag_size):
    """
    For a core containing multiple attachment points not only the Smiles needs to be canonicalized
    but also the attachment points. Attachment points are numbered according to their rank in a canonical
    ranking
    :param core: core smiles
    :param frags: list of substituent Smiles
    :param max_frag_size: maximum fragment size
    :return: core-Smiles, substituents-Smiles
    """

    frags = []
    coreFound = False
    for f in frags0:
        if not coreFound and f == core:
            coreFound = True
        else:
            frags.append(f)
            continue
    # frags = list(filter(lambda x: x != core, frags))

    if any(map(lambda x: count_heavy_atoms(x) > max_frag_size, frags)):
        return None, None
    if len(frags) == 1:
        # In case of a single substitution no further canonicalization needed
        return core, frags[0]
    core_mol = RWMol(Chem.MolFromSmiles(core))
    # Reset all map numbers for dummy atoms to 0, save original value
    saved_site_numbers = dict()
    for atom in core_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            saved_site_numbers[atom.GetIdx()] = atom.GetAtomMapNum()
            atom.SetAtomMapNum(0)
    # All map numbers are 0, so canonical ranking is not influenced by them
    atom_ranks = Chem.CanonicalRankAtoms(core_mol)
    order = ranks_to_order(atom_ranks)
    count = 0
    # renumber map numbers according to canonical order
    rsite_map = {}
    for idx in order:
        atom: Atom = core_mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 0:
            count += 1
            rsite_map[saved_site_numbers[atom.GetIdx()]] = count
            atom.SetAtomMapNum(count)
    core_smi = Chem.MolToSmiles(core_mol)
    # Renumber fragments accordingly
    substituents = ".".join(map(lambda x: renumber_r_atoms(x, rsite_map), frags))
    return core_smi, substituents


def canonicalize_mmp_cut(mmp_fragment: str, frags: List[str]):
    """
    For a frame containing multiple attachment points not only the Smiles needs to be canonicalized
    but also the attachment points. Attachment points are numbered according to their rank in a canonical
    ranking
    :param mmp_fragment: fragment smiles
    :param frags: list of all fragment Smiles
    :return: core-Smiles, substituents-Smiles
    """
    frags = list(filter(lambda x: x != mmp_fragment, frags))
    if len(frags) == 1:
        # In case of a single substitution no further canonicalization needed
        return frags[0], mmp_fragment
    frame = ".".join(frags)
    frame_mol = RWMol(Chem.MolFromSmiles(frame))
    # Reset all map numbers for dummy atoms to 0, save original value
    saved_site_numbers = dict()
    for atom in frame_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            saved_site_numbers[atom.GetIdx()] = atom.GetAtomMapNum()
            atom.SetAtomMapNum(0)
    # All map numbers are 0, so canonical ranking is not influenced by them
    atom_ranks = Chem.CanonicalRankAtoms(frame_mol)
    order = ranks_to_order(atom_ranks)
    count = 0
    # renumber map numbers according to canonical order
    rsite_map = {}
    for idx in order:
        atom: Atom = frame_mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 0:
            count += 1
            rsite_map[saved_site_numbers[atom.GetIdx()]] = count
            atom.SetAtomMapNum(count)
    frame_smi = Chem.MolToSmiles(frame_mol)
    # Renumber fragments accordingly
    fragment_smi = renumber_r_atoms(mmp_fragment, rsite_map)
    return frame_smi, fragment_smi


def fragment(smi: str, max_cuts: int, mode="MMP", get_removable_bonds=get_acyclic_single_bonds, min_rel_core_size=0.666,
             max_frag_size=13, max_time=100) -> Dict[str, str]:
    """
    Recursively fragment a molecule by cutting up to max_cuts single acyclic bonds.

    >>> fragment("Oc1cc(N)ccc1C",3,"CCR")
    {'Cc1ccc(N)cc1[*:1]': 'O[*:1]', 'Cc1ccc([*:1])cc1[*:2]': 'N[*:1].O[*:2]', 'c1cc([*:2])c([*:1])cc1[*:3]': 'C[*:2].N[*:3].O[*:1]', 'Nc1ccc([*:2])c([*:1])c1': 'C[*:2].O[*:1]', 'Cc1ccc([*:1])cc1O': 'N[*:1]', 'Oc1cc([*:2])ccc1[*:1]': 'C[*:1].N[*:2]', 'Nc1ccc([*:1])c(O)c1': 'C[*:1]'}

    :param smi: Smiles of molecule to fragment
    :param max_cuts: Max. number of non-hydrogen cuts allowed
    :param mode: Either "MMP" or "CCR".
                 "MMP" will produce cores corresponding to frames that consist of one to max_cuts fragments
                       while the substituent is a single fragment.
                 "CCR" will produce cores consisting of a single scaffold with attachment points
                       while substituents can be 1 to max_cuts fragments.
    :param get_removable_bonds: get_acyclic_single_bonds or get_synthesizable bonds.
                                Function detemining the list of cuttable bonds
    :param min_rel_core_size: minimum relative size required of core 0<=min_rel_core_size<1,
                              e.g., 0.5 requires that the core contains at least half the heavy atoms of the molecule
    :param max_frag_size: maximum fragment size
    :param max_time: time limit in seconds for fragmentation. Large molecules with many cuttable bonds can take
                     a long time. If the limit is exceeded None is returned
    :return: dictionary where the key is the smiles of a core and the value
             is the smiles of the fragments as disconnected components. Attachment points are represented by dummy
             atoms using atom map numbers
    """
    time_limit = time.time() + max_time

    def retrieve_core_and_substituents(mol: RWMol):
        frags = Chem.MolToSmiles(mol).split(".")
        frags = list(map(lambda f: Chem.MolToSmiles(Chem.MolFromSmiles(f)), frags))
        counts = list(map(count_heavy_atoms_and_substitution_sites, frags))
        mol_size = sum(map(itemgetter(0), counts))
        min_core_size = mol_size * min_rel_core_size
        max_mmp_frag_size = min(mol_size - min_core_size, max_frag_size)
        cut_count = len(counts) - 1
        good_core = False
        if mode == "MMP":
            for (ha, rg), frag in zip(counts, frags):
                if rg == cut_count and ha <= max_mmp_frag_size:
                    core, fragment = canonicalize_mmp_cut(frag, frags)
                    cuts[core] = fragment
                    # print(core, substituents)
                    good_core = True
        else:
            for (ha, rg), frag in zip(counts, frags):
                if rg == cut_count and ha >= min_core_size:
                    core, substituents = canonicalize_ccr_cut(frag, frags, max_frag_size)
                    if core:
                        cuts[core] = substituents
                        # print(core, substituents)
                        good_core = True
        return good_core

    def recurse(start_idx: int, cut_number: int):
        """

        :param start_idx: index of bond to remove
        :param cut_number: cut number to assign to current cut, increases with each additional cut
        :return: None
        """
        if time.time() > time_limit:
            fail[0] = True
            return
        for idx in range(start_idx, len(removable_bonds)):
            bond_atoms = removable_bonds[idx]
            cuttable = CuttableBond(mol, bond_atoms, cut_number)
            cuttable.cut_bond()
            # frags = Chem.MolToSmiles(mol).split(".")

            good_core = retrieve_core_and_substituents(
                mol)  # Call also takes care of saving current cut, if it is valid
            if (good_core or mode == "MMP") and cut_number < max_cuts:
                recurse(idx + 1, cut_number + 1)
            cuttable.undo_cut()

    mol = RWMol(Chem.MolFromSmiles(smi))
    removable_bonds = get_removable_bonds(mol)
    fail = [False]
    cuts: Dict[str, str] = dict()
    recurse(0, 1)
    if fail[0]:
        # Failure occurs when time limit is exceeded
        # return number of cuts generated so far for statistical purposes
        return len(cuts)
    # return valid framgentations as core/substituents dictionary in case of success
    return cuts


def aligned_pprint(adict, key_header, value_header, smiles=None):
    if smiles:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    if adict:
        n = max(map(len, adict.keys()))
        print("{key:>{n}}    {value}".format(key=key_header, value=value_header, n=n))
        for key in sorted(adict.keys(), key=len, reverse=True):
            print("{key:>{n}}    {value}".format(key=key, value=adict[key], n=n))
            mol = reconstruct_molecule(key, adict[key])
            if smiles and smiles != Chem.MolToSmiles(mol):
                print("ERROR:")
                print(smiles)
                print(Chem.MolToSmiles(mol))


def reconstruct_molecule(core: str, substituents: Union[str, Dict[int, str]]) -> Optional[Mol]:
    """
    Reassamble a molecule from kits core and fragment smiles
    :param core: Core Smiles with substitution sites
    :param substituents: A smiles containing fragments wiht the appropriate substitution sites
    :return: reconstructed molecules as Mol object
    """
    if type(substituents) == dict:
        substituents = ".".join(to_specific_rsite_map(substituents).values())
    mol = Chem.MolFromSmiles(core + "." + substituents) if substituents else Chem.MolFromSmiles(core)
    if not mol:
        logger.warning("Illegal Smiles: ", core + "." + substituents)
        return None
    mol = Chem.RWMol(mol)
    rsite_map = defaultdict(list)
    at: Atom
    for at in mol.GetAtoms():
        if at.GetAtomicNum() == 0:
            key = at.GetAtomMapNum()
            rsite_map[key].append(at)
    for pairs in rsite_map.values():
        if len(pairs) != 2:
            logger.error("R-sites not properly paired in {}".format(core + "." + substituents))
            return None
    for idx, pairs in rsite_map.items():
        if pairs[0].GetDegree() == 0 or pairs[1].GetDegree() == 0:  # hydrogen cut
            atom = pairs[0].GetNeighbors()[0] if pairs[0].GetDegree() != 0 else pairs[1].GetNeighbors()[0]
            mol.RemoveAtom(pairs[0].GetIdx())
            mol.RemoveAtom(pairs[1].GetIdx())
            protonate_atom(atom)
        else:
            CuttableBond(mol, None, idx, tuple(pairs)).undo_cut()
    return mol


if __name__ == '__main__':
    print("CCR MODE")
    smi = "Oc1cc(N)ccc1C"
    fragmentations = fragment(smi, 3, "CCR")
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents")
    smi = "CN1C(=NC(C1=O)(c2ccccc2)c3ccccc3)N"
    fragmentations = fragment(smi, 3, "CCR")
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents")
    smi = "CC[C@H](C)[C@@H]1C(=O)N[C@@H](C(=O)N1[C@H](C2=COC(=N2)C)C(=O)N3CCOCC3)C4CC5=CC=CC=C5C4"
    fragmentations = fragment(smi, 3, "CCR")
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents", smi)

    print("MMP MODE")
    smi = "Oc1cc(N)ccc1C"
    fragmentations = fragment(smi, 3, "MMP")
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents")
    smi = "CN1C(=NC(C1=O)(c2ccccc2)c3ccccc3)N"
    fragmentations = fragment(smi, 3, "MMP")
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents")
    smi = "Cc1cc2c(cc1Oc3ccc(o3)C(=O)Nc4c(nc(nc4OC)NCCCN5CCOCC5)OC)C(CC2)(C)C"
    fragmentations = fragment(smi, 3, "MMP", get_synthesizable_single_bonds, 0.5)
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents")
    fragmentations = fragment(smi, 3, "MMP", get_acyclic_single_bonds, 0.5)
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents")
    fragmentations = fragment(smi, 5, "CCR", get_synthesizable_single_bonds, 0.5)
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents")
    fragmentations = fragment(smi, 5, "CCR", get_acyclic_single_bonds, 0.5)
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents")
    smi = "CC[C@H](C)[C@@H]1C(=O)N[C@@H](C(=O)N1[C@H](C2=COC(=N2)C)C(=O)N3CCOCC3)C4CC5=CC=CC=C5C4"
    fragmentations = fragment(smi, 3, "CCR")
    print("molecule:", smi)
    aligned_pprint(fragmentations, "core", "substituents", smi)

