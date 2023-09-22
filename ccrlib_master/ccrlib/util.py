import logging
import re
from collections import defaultdict
from typing import NamedTuple, Union, List, Tuple, Dict, Optional

from rdkit import Chem, Chem as Chem
from rdkit.Chem import Mol, Atom, RWMol, BondType


logger = logging.getLogger(__name__)

periodic_table = Chem.GetPeriodicTable()

class _objectview:
    def __init__(self, d):
        self.__dict__ = d


class NamedSmiles(NamedTuple):
    smiles: str
    name: str


class NamedSmilesWithCore(NamedTuple):
    smiles: str
    name: str
    core: str


class FragmentationWithKey(NamedTuple):
    key: str
    substituents: str
    name: str
    core: str


Fragmentation = Union[NamedSmiles, NamedSmilesWithCore]


# Methods for SARMs


def replace_element(smi_or_mol,src=0,repl=85):
    if isinstance(smi_or_mol,str):
        src = "["+periodic_table.GetElementSymbol(src)
        repl = "["+periodic_table.GetElementSymbol(repl)
        return smi_or_mol.replace(src,repl)
    else:
        mol = Mol(smi_or_mol)
        for a in mol.GetAtoms():
            if a.GetAtomicNum()==src:
                a.SetAtomicNum(repl)
        mol.UpdatePropertyCache()
        return mol
    pass


def replace_substitution_sites_with_dummy_element(smi_or_mol:Union[str,Mol],el=85):
    # 'Good' dummy atoms with valence 1: 87 Fr,85 At,55 Cs, 37 Rb
    return replace_element(smi_or_mol,0,el)


def replace_dummy_element_with_substitution_sites(smi_or_mol:Union[str,Mol],el=85):
    # 'Good' dummy atoms with valence 1: 87 Fr,85 At,55 Cs, 37 Rb
    return replace_element(smi_or_mol,el,0)


def rsite_replaced_core_supplier(cores: List[str],dummy_el=85):
    for core in cores:
        rsmi = replace_element(core,0,dummy_el)
        mol = Chem.MolFromSmiles(rsmi)
        if mol is None:
            raise RuntimeError(f"Bad smiles: {rsmi}")
        mol.SetProp("_Name",core)
        yield mol

# End: Methods for SARMs

"""
# Virtual core modifications
class VirtualCoreToCores:
    def __init__(self, virtualize_core: Callable[[Union[str, Mol]], (Mol, Mol)]):
        # self.core_set=dict()
        self.vcores = dict()
        self.virtualize_core = virtualize_core

    def register_fragmentation(self, core: str, name: NamedSmiles) -> str:
        core_mol, vcore_mol = self.virtualize_core(core)
        virtual_core = Chem.MolToSmiles(vcore_mol)
        # core = self.core_set.setdefault(core,core) # Internalization (could use sys.intern)
        # virtual_core = self.core_set.setdefault(virtual_core,virtual_core) # Internalization (could use sys.intern)
        mol_dict = self.vcores.setdefault(virtual_core, dict())
        mol_dict[name] = core
        return virtual_core
"""

# unused for now
# def add_substitution_sites_to_real_core(vcore: str, rcore: str, virtualize_core: CoreAbstraction) -> str:
#     _, vmol, vmap = get_attachment_points_and_remove_r_atoms(vcore)
#     _, rmol, rmap = get_attachment_points_and_remove_r_atoms(rcore)
#     rmol, rvmol = virtualize_core(rmol)
#     vranks = Chem.CanonicalRankAtoms(vmol)
#     rranks = Chem.CanonicalRankAtoms(rvmol)
#     invranks = {rk: idx for idx, rk in enumerate(rranks)}
#     rmap = {}
#     for idx, ri in vmap.items():
#         rmap[invranks[vranks[idx]]] = ri
#     return add_r_atoms(rmol, rmap)

### Generic helper functions



def as_dict(collection,key_index,value_index = None):
    d = dict()
    if value_index:
        for entry in collection:
            d[entry[key_index]] = entry[value_index]
    else:
        for entry in collection:
            d[entry[key_index]] = entry
    return d

# noinspection PyUnusedLocal
def _true_fct(x):
    return True



# noinspection PyPep8Naming
class _identity_dict(dict):
    def __missing__(self, key):
        return key


def head(c):
    """
    Return first element of an iterable collection
    """
    return next(c.__iter__())


def star_to_r(smiles: str) -> str:
    """
    Convert RDKit's [*:d] to OpenEye's [Rd] in Smiles
    """
    return re.sub(r"\[\*:(\d+)]", r"[R\1]", smiles)


def r_to_star(smiles: str) -> str:
    """
    Convert OpenEye's [Rd] to RDKit#s [*:d] in Smiles
    """
    return re.sub(r"\[R(\d+)]", r"[*:\1]", smiles)



# *** Mol-handling helper functions ***

def get_faulty_explicit_valence(atom: Atom) -> int:
    """
    In the "faulty" explicit valence calculation a pair of aromatic bonds
    adds 2 to the explicit valence (instead of 3).
    :param atom:
    :return:
    """
    v = atom.GetNumExplicitHs()
    v += sum([int(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds()])
    return int(v)

# This simple version does not handle tradeoff between hydrogens and formal charges

def hydrogenize_core_legacy(core: str) -> str:
    " Remove substitution sites of the form[*:d] in Smiles by substituting them with an hydrogens"
    hcore = re.sub(r"\[\*:\d+\]", "[H]", core)
    mol = Chem.MolFromSmiles(hcore)
    return Chem.MolToSmiles(mol)


def hydrogenize_core(core: str) -> str:
    """
    Remove substitution sites of the form [*:d] in Smiles
    by substituting them with an hydrogens
    """
    _, mol, _ = get_attachment_points_and_remove_r_atoms(core)
    return Chem.MolToSmiles(mol)


def protonate_atom(atom: Atom) -> None:
    fc = atom.GetFormalCharge()
    group_CNOF = periodic_table.GetNOuterElecs(atom.GetAtomicNum()) >= 4
    # First try to adjust formal charge to become more neutral
    if fc > 0 and group_CNOF:
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
    elif fc < 0 and not group_CNOF:
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
    else:
        # If fc adjustment fails add a hydrogen
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
    atom.UpdatePropertyCache()
    return


def deprotonate_atom(atom: Atom) -> None:
    if atom.GetNumImplicitHs() == 0:
        # Cannot compensate by adjusting implicit Hs
        # Can we adjust by explicit Hs
        if atom.GetNumExplicitHs() > 0:  # Yes
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
        else:  # No, adjust formal charge instead
            if periodic_table.GetNOuterElecs(atom.GetAtomicNum()) >= 4:
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)
            else:
                atom.SetFormalCharge(atom.GetFormalCharge() - 1)
    atom.UpdatePropertyCache()


def get_attachment_points_and_remove_r_atoms(core: str) -> Tuple[str, Mol, Dict[int, List[int]]]:
    """
    Identify the attachment atoms of R-groups
    associate the r-site indices with the attachment atoms
    remove attachment atoms from core

    :param core:
    :return: original core smiles, core with r-atoms removed as mol,
            dict. associating atom attachment points with rsite indices
    """
    mol = RWMol(Chem.MolFromSmiles(core))
    sub_sites0: Dict[int, List[int]] = defaultdict(list)
    to_delete = []
    atom: Atom
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            ap = atom.GetNeighbors()[0]
            sub_sites0[ap.GetIdx()].append((atom.GetAtomMapNum()))
            to_delete.append(atom)
    sub_sites0: Dict[Atom, List[int]] = {mol.GetAtomWithIdx(k): v for k, v in sub_sites0.items()}
    for atom in to_delete:
        mol.RemoveAtom(atom.GetIdx())
    for atom in sub_sites0:
        numHs = len(sub_sites0[atom])
        # Correct r-site removal by adjusting formal charges or adding hydrogens
        while numHs > 0:
            protonate_atom(atom)
            numHs -= 1
    sub_sites1: Dict[int, List[int]] = {a.GetIdx(): map_num for a, map_num in sub_sites0.items()}
    atom_ranks: List[int] = Chem.CanonicalRankAtoms(mol)
    order = [None] * len(atom_ranks)
    for (i, r) in enumerate(atom_ranks):
        order[r] = i
    mol: Mol = Chem.RenumberAtoms(mol, order)
    sub_sites2: Dict[int, List[int]] = dict()
    for (nidx, oidx) in enumerate(order):
        if oidx in sub_sites1:
            sub_sites2[nidx] = sub_sites1[oidx]
    return core, mol, sub_sites2


def add_r_atoms(mol0: Mol, sub_sites: Dict[int, List[int]]) -> RWMol:
    mol = RWMol(mol0)
    for idx in sorted(sub_sites.keys()):
        for r_num in sub_sites[idx]:
            r_atom = Atom(0)
            r_atom.SetAtomMapNum(r_num)
            r_idx = mol.AddAtom(r_atom)
            mol.AddBond(r_idx, idx, BondType.SINGLE)
            atom: Atom = mol.GetAtomWithIdx(idx)
            try:
                deprotonate_atom(atom)
            except Chem.AtomValenceException as e:
                logger.error(f"AtomValenceException: {mol0} at {idx} for {str(sub_sites)}")
                raise
    return mol


def get_rsite(smiles: str) -> int:
    return int(re.search(r"\[\*:(\d+)]", smiles).group(1))


def get_rsites(smiles: str) -> List[int]:
    return [int(m.group(1)) for m in re.finditer(r"\[\*:(\d+)]", smiles)]


def to_rsite_map(smiles: str) -> Dict[int, str]:
    rsites = [int(m.group(1)) for m in re.finditer(r"\[\*:(\d+)]", smiles)]
    return dict(zip(rsites, smiles.split(".")))


def to_generic_rsite_map(smiles: str) -> Dict[int, str]:
    rmap = to_rsite_map(smiles)
    return {k: re.sub(r"\[\*:(\d+)]", "*", v) for k, v in rmap.items()}


def to_specific_rsite_map(frag_map: Dict[int, str]) -> Dict[int, str]:
    return {k: re.sub(r"\*", f"[*:{k}]", v) for k, v in frag_map.items()}


def generic_rsite_map_to_fragment_smiles(frag_map: Dict[int, str]) -> str:
    rmap = to_specific_rsite_map(frag_map)
    return ".".join(rmap[k] for k in sorted(rmap))


def swap_rsites(core: str, a: int, b: int) -> str:
    # t = re.sub(r"\[\*:{}\]".format(a),"##",core)
    # t = re.sub(r"\[\*:{}\]".format(b),r"\[\*:{}\]".format(a),t)
    # return re.sub("##","\[\*:{}\]".format(b),t)
    remap = _identity_dict()
    remap.update({a: b, b: a})
    return renumber_r_atoms(core, remap)


def renumber_r_atoms(smi: str, remap):
    """
    Remaps the substitution site map indices
    :param smi: Smiles containing dummy atoms with map indices representing attachment points
    :param remap: dictionary mapping old indices to new indices
    :return: Smiles with remapped indices
    """

    def repl(m):
        return "[*:%d]" % remap[int(m.group(1))]

    return re.sub(r"\[\*:(\d+)\]", repl, smi)


