from typing import Callable, Union, Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Mol, BondType

CoreAbstraction = Callable[[Union[str, Mol]], Tuple[Mol, Mol]]
CoreAbstraction = Callable

def oxygen_sulfur_halogen_abstraction(core: Union[str, Mol]) -> (Mol, Mol):
    """
    The function should return the core (as a Mol object) and the virtual core as a mol object
    The virtual core is a "generalized core" where corresponding atoms are not distinguished.
    This is realized by representing corresponding atoms with a single atom type.
    Atom indices are unchanged between core and vcore.
    :param core: The core as a Smiles string or as Mol object
    :return core, vcore: core is the core as Mol object and vcore is th virtualized core
    """
    if isinstance(core, str):
        core = Chem.MolFromSmiles(core)
        if core is None:
            return None, None
    vcore = Mol(core)
    for a in vcore.GetAtoms():
        # Valence 2 sulfur and oxygens are considered analogous and interchangable in a core
        an = a.GetAtomicNum()
        if an == 16 and a.GetTotalValence() == 2:
            a.SetAtomicNum(8)
        elif an in [9,17,35,53]:
            # Halogens
            a.SetAtomicNum(9)
    return core, vcore

def oxygen_sulfur_abstraction(core: Union[str, Mol]) -> (Mol, Mol):
    """
    The function should return the core (as a Mol object) and the virtual core as a mol object
    The virtual core is a "generalized core" where corresponding atoms are not distinguished.
    This is realized by representing corresponding atoms with a single atom type.
    Atom indices are unchanged between core and vcore.
    :param core: The core as a Smiles string or as Mol object
    :return core, vcore: core is the core as Mol object and vcore is th virtualized core
    """
    if isinstance(core, str):
        core = Chem.MolFromSmiles(core)
        if core is None:
            return None, None
    vcore = Mol(core)
    for a in vcore.GetAtoms():
        # Valence 2 sulfur and oxygens are considered analogous and interchangable in a core
        an = a.GetAtomicNum()
        if an == 16 and a.GetTotalValence() == 2:
            a.SetAtomicNum(8)
    return core, vcore

## Only replace atoms with atoms with comparable valence
def carbon_scaffold_abstraction(core: Union[str, Mol]) -> (Mol, Mol):
    raise NotImplementedError("carbon_scaffold_abstraction not implemented")
    if isinstance(core, str):
        core = Chem.MolFromSmiles(core)
        if core is None:
            return None, None
    vcore = Mol(core)
    for a in vcore.GetAtoms():
        # Valence 2 sulfur and oxygens are considered analogous and interchangable in a core
        if a.GetAtomicNum() > 0:
            a.SetAtomicNum(6)
            a.SetNumExplicitHs(0)
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
    for b in vcore.GetBonds():
        b.SetBondType(BondType.SINGLE) # Works only if no double-bond substitutions are allowed
        b.SetIsAromatic(False)
    vcore.UpdatePropertyCache()
    return core,vcore

