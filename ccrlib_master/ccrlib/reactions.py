class IsSynthesizable:
    """
    :param mol_1: molecule 1
    :param mol_2: molecule 2
    :param a1: atom 1
    :param a2: atom 2
    :param swap: bool
    :param track_reactions: bool
    :param allow_multiaromatic_N_O_S: bool

    Fragmentation scenario:
    - the class is used to detect cutable bonds ==> mol_1 == mol_2 (the molecule to be cut)
    - the class needs to be initialized two separate times - once with the swap param as False and once as True
    Enumeration scenario:
    - the class is used to detect possible bond formation ==> mol_1 != mol_2 (these are molecules to be connected)
    - the class needs to be initialized two separate times - once with the swap param as False and once as True
    - Note that in the bond formation scenario mol_1 and mol_2 must be different (not identical) rdkit Mol OBJECTS!
    (even if bond formation between two identical molecules molecules is attempted!, they have to be assigned to different )

    """

    def __init__(self, mol_1, mol_2, a1, a2, swap=False, track_reactions=True, allow_multiaromatic_N_O_S=False):
        assert isinstance(swap, bool)
        self.swap = swap
        self.mol_1 = mol_1 if self.swap is False else mol_2
        self.mol_2 = mol_2 if self.swap is False else mol_1
        self.a1 = a1 if self.swap is False else a2
        self.a2 = a2 if self.swap is False else a1
        self.track_reactions = track_reactions
        self.allow_multiaromatic_N_O_S = allow_multiaromatic_N_O_S

    def _record_reaction(self, reaction_name):
        return [reaction_name, self.mol_1, self.mol_2, self.a1.GetIdx(), self.a2.GetIdx()]

    def _get_a1_nbrs(self):
        return [x for x in self.a1.GetNeighbors() if
                (self.mol_1 is not self.mol_2 or x.GetIdx() != self.a2.GetIdx()) and x.GetAtomicNum() != 0]

    def _get_a2_nbrs(self):
        return [x for x in self.a2.GetNeighbors() if
                (self.mol_1 is not self.mol_2 or x.GetIdx() != self.a1.GetIdx()) and x.GetAtomicNum() != 0]

    def is_fragmentation(self):
        if self.mol_1 == self.mol_2:
            return True
        return False

    def is_bond_formation(self):
        if self.mol_1 != self.mol_2:
            return True
        return False

    def is_single_exocyclic_bond(self):
        if self.is_fragmentation():
            bond = self.mol_1.GetBondBetweenAtoms(self.a1.GetIdx(), self.a2.GetIdx())
            if str(bond.GetBondType()) == "SINGLE" and not bond.IsInRing():
                return True
            return False
        return False

    @staticmethod
    def reactions():
        return {y: x for x, y in IsSynthesizable.__dict__.items() if
                not x.startswith("_") and not x.startswith("is") and x not in ("reactions", "detect")}

    def detect(self):
        """returns a list of reaction details or None if track_reactions is True
            returns True or False if track_reactions is False"""

        if self.track_reactions is True:

            # consider only single exocyclic bonds for fragmentation
            if self.is_fragmentation() and not self.is_single_exocyclic_bond():
                return None

            # homoatomic bonds
            if self.a1.GetAtomicNum() == self.a2.GetAtomicNum():

                if self.a1.GetAtomicNum() == 6:  # C-C single bond

                    if self.suzuki() is True:
                        return self._record_reaction("suzuki")  # RECAP
                    if self.negishi() is True:
                        return self._record_reaction("negishi")
                    if self.stille() is True:
                        return self._record_reaction("stille")
                    if self.sonogashira() is True:
                        return self._record_reaction("sonogashira")
                    if self.heck() is True:
                        return self._record_reaction("heck")
                    if self.grignard_carbonyl() is True:
                        return self._record_reaction("grignard_carbonyl")
                    if self.grignard_alcohol() is True:
                        return self._record_reaction("grignard_alcohol")

                elif self.a1.GetAtomicNum() == 16:  # S-S single bond

                    if self.disulfide() is True:
                        return self._record_reaction("disulfide")  # RECAP

                else:
                    return None  # any other atom except for C or S

            # heteroatomic bonds
            else:

                # RECAP rules (method calling order should be maintained as it is)
                if self.aroN_aliC() is True:
                    return self._record_reaction("aroN_aliC")
                if self.sulfonamide() is True:
                    return self._record_reaction("sulfonamide")
                if self.lactam() is True:
                    return self._record_reaction("lactam")
                if self.amide(thio=False) is True:
                    return self._record_reaction("amide")
                if self.amide(thio=True) is True:
                    return self._record_reaction("thioamide")
                if self.ester(thio=False) is True:
                    return self._record_reaction("ester")
                if self.ester(thio=True) is True:
                    return self._record_reaction("thioester")
                if self.amine() is True:
                    return self._record_reaction("amine")
                if self.ether(thio=False) is True:
                    return self._record_reaction("ether")
                if self.ether(thio=True) is True:
                    return self._record_reaction("thioether")

                # additional rules
                if self.aroN_aroC() is True:
                    return self._record_reaction("aroN_aroC")
                if self.mitsunobu_imide() is True:
                    return self._record_reaction("mitsunobu_imide")
                if self.mistunobu_sulfonamide() is True:
                    return self._record_reaction("mistunobu_sulfonamide")
                if self.piperidine_indole() is True:
                    return self._record_reaction("piperidine_indole")

                return None

        # no need to record reactions
        else:

            # consider only single exocyclic bonds for fragmentation
            if self.is_fragmentation() and not self.is_single_exocyclic_bond():
                return False

            # homoatomic bonds
            if self.a1.GetAtomicNum() == self.a2.GetAtomicNum():

                if self.a1.GetAtomicNum() == 6:  # C-C single bond
                    carbon_carbon = [self.suzuki(), self.negishi(), self.stille(), self.sonogashira(),
                                     self.heck(), self.grignard_carbonyl(), self.grignard_alcohol()]
                    if any(carbon_carbon):
                        return True

                elif self.a1.GetAtomicNum() == 16:  # S-S single bond
                    sulfur_sulfur = [self.disulfide()]
                    if any(sulfur_sulfur):
                        return True
                else:
                    return False  # any other atom except for C or S

            # heteroatomic bonds
            else:
                heteroatomic_recap = [self.aroN_aliC(), self.sulfonamide(), self.lactam(),
                                      self.amide(thio=False), self.amide(thio=True),
                                      self.ester(thio=False), self.ester(thio=True), self.amine(),
                                      self.ether(thio=False), self.ether(thio=True)]
                heteroatomic_non_recap = [self.aroN_aroC(), self.mitsunobu_imide(), self.mistunobu_sulfonamide(),
                                          self.piperidine_indole()]

                if any(heteroatomic_recap + heteroatomic_non_recap):
                    return True

            return False

    # RECAP methods

    def suzuki(self):
        # aroC-aroC // "[c:1]-!@[c:2]>>[c:1][*].[*][c:2]" (Suzuki coupling)
        if self.a1.GetIsAromatic() and self.a2.GetIsAromatic():
            return True
        return False

    def disulfide(self):
        # Disulfide // [S:1]-!@[S:2]>>[S:1][*].[S:2][*]
        if not self.a1.GetIsAromatic() and not self.a2.GetIsAromatic() and self.a1.GetDegree() == 2 and self.a2.GetDegree() == 2:
            return True
        return False

    def aroN_aliC(self):
        # aroN-aliC // [n;+0:1]-!@[C:2]>>[n:1][*].[C:2][*]
        if self.a1.GetAtomicNum() == 6 and not self.a1.GetIsAromatic() and self.a2.GetAtomicNum() == 7 and self.a2.GetIsAromatic():
            return True
        return False

    def sulfonamide(self):
        # Sulfonamide
        if self.a1.GetAtomicNum() == 16 and self.a2.GetAtomicNum() == 7:
            oxygen_count = 0
            for a in self.a1.GetNeighbors():
                if a.GetAtomicNum() == 8 and self.mol_1.GetBondBetweenAtoms(a.GetIdx(),
                                                                            self.a1.GetIdx()).GetBondTypeAsDouble() == 2.0:
                    oxygen_count += 1
            if oxygen_count >= 2:
                return True
        return False

    def lactam(self):
        #  Lactam // "[O:3]=[C:4]-@[N;+0:1]-!@[C:2]>>[O:3]=[C:4]-[N:1][*].[C:2][*]"
        if self.a1.GetAtomicNum() == 6 and not self.a1.GetIsAromatic() and not self.a1.IsInRing() and \
                self.a2.GetAtomicNum() == 7 and not self.a2.GetIsAromatic() and self.a2.IsInRing():
            for a in self.a2.GetNeighbors():
                if not a.GetAtomicNum() == 6 or not a.IsInRing():
                    continue
                else:
                    for an in a.GetNeighbors():
                        if an.GetAtomicNum() == 8 and \
                                self.mol_2.GetBondBetweenAtoms(an.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2.0:
                            return True
        return False

    def amide(self, thio=False):
        # Amide/Thioamide // "[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>[*][C:1]=[O:2].[*][#7:3]"
        heavy_nbr = 8 if thio is False else 16
        if self.a1.GetAtomicNum() == 6 and self.a2.GetAtomicNum() == 7:
            for a in self.a1.GetNeighbors():
                if self.mol_1.GetBondBetweenAtoms(a.GetIdx(),
                                                  self.a1.GetIdx()).GetBondTypeAsDouble() == 2.0 and a.GetAtomicNum() == heavy_nbr:
                    return True
        return False

    def ester(self, thio=False):
        # Ester/Thioester // [C:1](=!@[O:2])!@[O;+0:3]>>[*][C:1]=[O:2].[O:3][*]
        heavy_nbr = 8 if thio is False else 16
        if self.a1.GetAtomicNum() == 6 and self.a2.GetAtomicNum() == heavy_nbr and self.a2.GetDegree() == 2:
            for a in self.a1.GetNeighbors():
                if self.mol_1.GetBondBetweenAtoms(a.GetIdx(),
                                                  self.a1.GetIdx()).GetBondTypeAsDouble() == 2.0 and a.GetAtomicNum() in (
                        8, 16):
                    return True
        return False

    def amine(self):
        # Amine // "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*:1])-!@[*:2]>>[*][*:1].[*:2][*]"
        if self.a1.GetAtomicNum() == 6 and self.a2.GetAtomicNum() == 7 and not self.a2.GetIsAromatic():

            # Do not allow not alkylated amino groups ==> -NH2
            if self.a2.GetDegree() < 2:
                return False

            aro_count = 0
            for a in self.a2.GetNeighbors():
                # loop over non R atoms
                if a.GetAtomicNum() == 0:
                    continue
                if not a.GetAtomicNum() == 6:  # do not allow NO2, NSO1...
                    return False
                else:
                    # do not allow amines exclusively between aromatic C atoms
                    if not self.allow_multiaromatic_N_O_S:
                        if a.GetIsAromatic() and self.a1.GetIsAromatic():
                            aro_count += 1
                        if aro_count == 2:
                            return False
                    for an in a.GetNeighbors():
                        if self.mol_2.GetBondBetweenAtoms(an.GetIdx(),
                                                          a.GetIdx()).GetBondTypeAsDouble() == 2 and an.GetAtomicNum() in (
                                8, 16):
                            return False  # exclude Amide/Thioamide
            return True
        return False

    def ether(self, thio=False):
        # Ether/Thioether // "[#6:1]-!@[O;+0]-!@[#6:2]>>[#6:1][*].[*][#6:2]"
        heavy_nbr = 8 if thio is False else 16
        if self.a1.GetAtomicNum() == 6 and self.a2.GetAtomicNum() == heavy_nbr:

            # Do not allow not alkylated alkohol/thio groups ==> -OH/-SH
            # allow postitively charged ether/thioether groups (if not permitted ==> change to a2.GetHvyDegree() != 2)
            if self.a2.GetDegree() < 2:  # HvyDegree is always at least 1 because of the R atom neighbor
                return False

            for a in self.a2.GetNeighbors():
                # loop over non R atoms
                if a.GetAtomicNum() == 0:
                    continue
                if not a.GetAtomicNum() == 6:  # do not allow any other heteroatoms (e.g. O-O, S-S, O-N, S-N ...)
                    return False
                else:
                    if not self.allow_multiaromatic_N_O_S:
                        if a.GetIsAromatic() and self.a1.GetIsAromatic():  # do not allow ether/thioether between two aromatic C atoms
                            return False
                    for an in a.GetNeighbors():
                        if self.mol_2.GetBondBetweenAtoms(an.GetIdx(),
                                                          a.GetIdx()).GetBondTypeAsDouble() == 2 and an.GetAtomicNum() in (
                                8, 16):
                            return False  # exclude Ester/Thionoester
            return True
        return False

    # Additional non-RECAP reactions

    def negishi(self):
        """C-C coupling"""
        # extract only the nbrs for each atom that are neither the other atom itself (in case of cutting) nor the R-atom (in case of combining)
        nbr_env_a1 = set([x.GetAtomicNum() for x in self._get_a1_nbrs()])
        nbr_env_a2 = set([x.GetAtomicNum() for x in self._get_a2_nbrs()])
        # no heteroatom nbrs of the Cs and no terminal methyl Cs allowed
        if len(nbr_env_a1) != 1 or nbr_env_a1.pop() != 6 or len(nbr_env_a2) != 1 or nbr_env_a2.pop() != 6:
            return False
        # aromatic C can be coupled only with an aliphatic C (sp3) or another aromatic C
        if self.a1.GetIsAromatic() and (self.a2.GetTotalDegree() != 4 or not self.a2.GetIsAromatic()):
            return False
        # alkenyl-/alkynyl C (sp2/sp) can be coupled only with an aliphatic C (sp3)
        elif self.a1.GetTotalDegree() in (2, 3) and self.a2.GetTotalDegree() != 4:
            return False
        # aliphatic (sp3) C can be coupled with any other C
        else:
            return True

    def stille(self):
        """Vinyl C - aromatic C or aroC-aroC (only the vinyl C - aromatic C implemented because the rest
        is already checked by the suzuku  coupling"""
        # vinyl root C and any aromatic C
        if self.a1.GetTotalDegree() == 3 and self.a1.GetTotalNumHs() == 1 and not self.a1.GetIsAromatic() \
                and self.a2.GetIsAromatic():
            # vinyl root C should have only 1 further nbr and it should be carbon connected with a olefinic double bond
            nbr_a1 = self._get_a1_nbrs()[0]
            if nbr_a1.GetAtomicNum() == 6 and nbr_a1.GetDegree() == 2 \
                    and self.mol_1.GetBondBetweenAtoms(nbr_a1.GetIdx(), self.a1.GetIdx()).GetBondTypeAsDouble() == 2.0:
                # the C nbr of the vinyl root C of the other side of the double bond should be connected only to 1 additional C of any type (according to SMIRKS)
                nbr_nbr_a1 = [x for x in nbr_a1.GetNeighbors() if x.GetIdx() != self.a1.GetIdx()][0]
                if nbr_nbr_a1.GetAtomicNum() == 6:
                    return True
            return False
        return False

    def sonogashira(self):
        """C-C bond between non-terminal sp C (alkyn) and vinyl or aryl C"""
        if self.a1.GetTotalDegree() == 3 and self.a2.GetTotalDegree() == 2:
            # validate vinyl or aromatic C
            for a1_nbr in self._get_a1_nbrs():
                if not a1_nbr.GetIsAromatic():  # vinyl C
                    if a1_nbr.GetAtomicNum() != 6:  # vinyl C
                        return False
                    for nbr in a1_nbr.GetNeighbors():
                        if nbr.GetAtomicNum() != 6:  # 2nd order nbrs need to be any carbon(s)
                            return False
            # validate alkyn
            for a2_nbr in self._get_a2_nbrs():  # there should be only one nbr (on the other side of the triple bond)
                if a2_nbr.GetAtomicNum() != 6:  # it must be C (not N e.g. as in nitrile)
                    return False
                # the direct nbr across the triple bond must have exactly 2 nbrs
                # (a2 itself and an additional C atom) because it should not be terminal alkyn
                for nbr in a2_nbr.GetNeighbors():  # should be only 1 2nd order nbr (sharing a single bond)
                    if nbr.GetIdx() == self.a2.GetIdx():
                        continue
                    elif nbr.GetAtomicNum() == 6:
                        return True
            return False
        return False

    def heck(self):
        """Olefinic C - olefinic/aromatic C

        if terminal vinyl C
        [#6;c,$(C(=O)O),$(C#N):3][#6:2]([#6:5])=[#6;H1;$([#6][#6]):1].[#6;$([#6]=[#6]),$(c:c):4][Cl,Br,I]>>[#6:4][#6;H0:1]=[#6:2]([#6:5])[#6:3]

        if non-terminal vinyl C
        [#6;c,$(C(=O)O),$(C#N):3][#6;H1:2]=[#6;H2:1].[#6;$([#6]=[#6]),$(c:c):4][Cl,Br,I]>>[#6:4]/[#6:1]=[#6:2]/[#6:3]

        """
        if self.a1.GetTotalDegree() == 3 and self.a1.GetTotalNumHs() in (
        0, 1) and not self.a1.GetIsAromatic() and self.a2.GetTotalDegree() == 3:

            # detect vinyl/aromatic C ==> according to SMART
            c_count = 0
            for nbr_a2 in self._get_a2_nbrs():
                # vinyl C
                if not self.a2.GetIsAromatic() and self.mol_2.GetBondBetweenAtoms(nbr_a2.GetIdx(),
                                                                                  self.a2.GetIdx()).GetBondTypeAsDouble() == 2.0 and nbr_a2.GetAtomicNum() != 6:
                    return False
                if nbr_a2.GetAtomicNum() == 6:
                    c_count += 1

            # must have at least one carbon nbr (vinyl or aromatic nbr)
            if c_count == 0:
                return False

            # detect secondary terminal or tertiary non-terminal vinyl C (must have only C heavy nbrs)
            nitril_count = 0
            o_count = 0
            aryl_count = 0
            for nbr_a1 in self._get_a1_nbrs():
                if nbr_a1.GetAtomicNum() != 6:
                    return False
                if self.mol_1.GetBondBetweenAtoms(nbr_a1.GetIdx(), self.a1.GetIdx()).GetBondTypeAsDouble() == 2.0:
                    # the vinyl C across the double bond has to be tertiary (3 C nbrs)
                    for n in nbr_a1.GetNeighbors():
                        if n.GetIdx() == self.a1.GetIdx():
                            continue
                        if n.GetAtomicNum() != 6:
                            return False
                        if n.GetIsAromatic():
                            aryl_count += 1
                        for nn in n.GetNeighbors():
                            if nn.GetAtomicNum() == 7 and self.mol_1.GetBondBetweenAtoms(n.GetIdx(),
                                                                                         nn.GetIdx()).GetBondTypeAsDouble() == 3.0:
                                nitril_count += 1
                            if nn.GetAtomicNum() == 8 and n.GetTotalDegree() == 3:
                                o_count += 1
            if aryl_count >= 1 or nitril_count >= 1 or o_count >= 2:
                return True

            return False
        return False

    def grignard_carbonyl(self):
        """C-C bond with ketone"""
        # non-terminal (not aldehyde) carbonyl (sp2) C
        # + any non-acyl C with minimum 1 other C nbr (any single, double, aromatic or triple bond) AND maximum 1 halogen nbr
        nbrs_a1 = self._get_a1_nbrs()
        nbrs_a2 = self._get_a2_nbrs()
        if self.a1.GetTotalDegree() == 3 and sum([x.GetAtomicNum() for x in nbrs_a1]) == 14 \
                and any([x.GetAtomicNum() == 6 for x in nbrs_a2]):
            # if a2 is alkyl C (sp3), maximum 1 halogen atom is allowed as direct nbr (according to reaction SMARTS)
            if self.a2.GetTotalDegree() == 4 and sum([x.GetAtomicNum() in (17, 35, 53) for x in nbrs_a2]) > 1:
                return False
            # if a2 is sp2 C, it cannot be acyl C (according to reaction SMARTS)
            if self.a2.GetTotalDegree() == 3 and any([x.GetAtomicNum() == 8 for x in nbrs_a2]):
                return False
            # detect carbonyl C
            acyl_count = 0
            c_count = 0
            for nbr_a1 in nbrs_a1:
                if nbr_a1.GetAtomicNum() == 8:  # 1 C=O bond allowed
                    if self.mol_1.GetBondBetweenAtoms(nbr_a1.GetIdx(), self.a1.GetIdx()).GetBondTypeAsDouble() == 2.0:
                        acyl_count += 1
                elif nbr_a1.GetAtomicNum() == 6:  # 1 C nbr of any type allowed
                    c_count += 1
                else:  # no other heteroatom nbrs alowed
                    return False
            if acyl_count == 1 and c_count == 1:
                return True

            return False
        return False

    def grignard_alcohol(self):
        """C-C bond with secondary/tertiary alcohol (from nucleophilic attack on aldehyde or ketone, respectively)"""
        # aliphatic secondary or tertiary C (from an aldehyde or ketone) i.e. without direct heteroatom nbrs other than the hydroxylic O
        # + any non-acyl C with minimum 1 other C nbr (any single, double, aromatic or triple bond) AND maximum 1 halogen nbr
        nbrs_a1 = self._get_a1_nbrs()
        nbrs_a2 = self._get_a2_nbrs()
        if self.a1.GetTotalDegree() == 4 and self.a1.GetTotalNumHs() in (0, 1) and all(
                [x.GetAtomicNum() in (6, 8) for x in nbrs_a1]) \
                and any([x.GetAtomicNum() == 6 for x in nbrs_a2]):
            # if a2 is alkyl C (sp3), maximum 1 halogen atom is allowed as direct nbr (according to reaction SMARTS)
            if self.a2.GetTotalDegree() == 4 and sum([x.GetAtomicNum() in (17, 35, 53) for x in nbrs_a2]) > 1:
                return False
            # if a2 is sp2 C, it cannot be acyl C (according to reaction SMARTS)
            if self.a2.GetTotalDegree() == 3 and any([x.GetAtomicNum() == 8 for x in nbrs_a2]):
                return False
            # detect secondary/tertiary alcohol
            o_count = 0
            c_count = 0
            for nbr_a1 in nbrs_a1:
                if nbr_a1.GetAtomicNum() == 8 and nbr_a1.GetTotalNumHs() == 1:  # max 1 O nbr allowed (from the tertiary -OH) that must be hydroxyl
                    o_count += 1
                elif nbr_a1.GetAtomicNum() == 6:  # 1 or 2 C nbr(s) allowed
                    c_count += 1
                else:  # no other direct heteroatom nbrs alowwed, so if a secondary alcohol, the last nbr must be H
                    return False
            # no need to check if the C-O bond is single because the total degree of the root C is 4 (sp3)
            if o_count == 1 and c_count in (1, 2):
                return True

            return False
        return False

    def aroN_aroC(self):
        """aroC-aroN
        [c:1]B(O)O.[nH1;+0;r5;!$(n[#6]=[O,S,N]);!$(n~n~n);!$(n~n~c~n);!$(n~c~n~n):2]>>[c:1][n:2]
        """
        # any aromatic C + tertiary aromatic N in a 5-membered ring with maximum one further aromatic N instide that ring
        if self.a1.GetAtomicNum() == 6 and self.a1.GetIsAromatic() \
                and self.a2.GetAtomicNum() == 7 and self.a2.GetIsAromatic() and self.a2.IsInRingSize(
            5) and self.a2.GetTotalDegree() == 3:
            # check aromatic ring atoms ==> maximum one further aromatic N (ortho or meta) accepted
            # no further comments in the paper on !$(n[#6]=[O,S,N]) ==> it seems to address the case of an ortho aromatic C with O,S,N attached (the enol case c(OH), c(SH), c(NH)), )
            # but it is represented in the SMARTS as the keto case i.e. any carbon (although it must be aromatic) in ortho with a double bond to O,S,N
            nbrs_ortho = self._get_a2_nbrs()
            nbrs_meta_putative = []
            nbrs_meta = []
            for nbr_ortho in nbrs_ortho:
                for n in nbr_ortho.GetNeighbors():
                    if n.GetIdx() == self.a2.GetIdx():  # skip nH
                        continue
                    if n.GetAtomicNum() in (7, 8,
                                            16) and not n.IsInRing() and not n.GetIsAromatic():  # aro check added because of Imine-enamine tautomers
                        return False
                    if n.IsInRingSize(5) and n.GetIsAromatic():
                        nbrs_meta_putative.append(n)

            for meta_nbr_1 in nbrs_meta_putative:
                for meta_nbr_2 in nbrs_meta_putative:
                    if meta_nbr_1.GetIdx() == meta_nbr_2.GetIdx():
                        continue
                    putative_bond = self.mol_2.GetBondBetweenAtoms(meta_nbr_1.GetIdx(), meta_nbr_2.GetIdx())
                    if putative_bond is not None and putative_bond.GetBondTypeAsDouble() == 1.5 and len(nbrs_meta) < 2:
                        nbrs_meta.extend([meta_nbr_1, meta_nbr_2])

            ring5_nbrs = nbrs_ortho + nbrs_meta
            # max. 2 aromatic Ns in the 5-ring (one is the connecting tartiary N + max. 1 more anywhere in the ring)
            if len(ring5_nbrs) == 4 and sum([x.GetAtomicNum() for x in ring5_nbrs]) in (24, 25):
                return True

            return False
        return False

    def mitsunobu_imide(self):
        """C-N bond"""
        # Educts: primary/secondary aliphatic C with secondary imide HN(C=O)C=O
        # Product: tertiary imide
        if self.a1.GetAtomicNum() == 6 and not self.a1.GetIsAromatic() and self.a1.GetTotalDegree() == 4 and self.a1.GetTotalNumHs() in (
                1, 2) \
                and self.a2.GetAtomicNum() == 7 and not self.a2.GetIsAromatic() and self.a2.GetTotalDegree() == 3 and self.a2.GetTotalNumHs() == 0:
            # extract only the nbrs for each atom that are neither the other atom itself (in case of cutting) nor the R-atom (in case of combining)
            nbrs_a1 = self._get_a1_nbrs()
            nbrs_a2 = self._get_a2_nbrs()
            # direct heteroatom nbrs of the aliphatic C are not alowwed
            if any([x.GetAtomicNum() != 6 for x in nbrs_a1]):
                return False
            # direct heteroatom nbrs of the imide N are not alowwed
            if any([x.GetAtomicNum() != 6 for x in nbrs_a2]):
                return False
            # nbrs of N must be acyl Cs (sp2 non-aromatic)
            if any([x.GetIsAromatic() for x in nbrs_a2]) or any([x.GetTotalDegree() != 3 for x in nbrs_a2]):
                return False
            # check for acyl C=O bonds
            acyl_count = 0
            for a2_nbr in nbrs_a2:
                for nbr in a2_nbr.GetNeighbors():
                    if nbr.GetIdx() == self.a2.GetIdx():  # skip the imide N
                        continue
                    if nbr.GetAtomicNum() == 8 and self.mol_2.GetBondBetweenAtoms(nbr.GetIdx(),
                                                                                  a2_nbr.GetIdx()).GetBondTypeAsDouble() == 2.0:
                        acyl_count += 1
            if acyl_count == 2:
                return True

            return False
        return False

    def mistunobu_sulfonamide(self):
        """C-N bond"""
        # Educts: primary/secondary aliphatic C with secondary sulfonamide HNS(=O)=O
        # Product: tertiary sulfonamide
        # extract only the nbrs for each atom that are neither the other atom itself (in case of cutting) nor the R-atom (in case of combining)
        nbrs_a1 = self._get_a1_nbrs()
        nbrs_a2 = self._get_a2_nbrs()

        # primary/secondary aliphatic C (direct nbrs can be only Cs of any kind except for a2 which is supposed to be N)
        # + aliphatic tertiary N
        if self.a1.GetAtomicNum() == 6 and not self.a1.GetIsAromatic() and self.a1.GetTotalDegree() == 4 \
                and self.a1.GetTotalNumHs() in (1, 2) and all([x.GetAtomicNum() == 6 for x in nbrs_a1]) \
                and self.a2.GetAtomicNum() == 7 and not self.a2.GetIsAromatic() and self.a2.GetTotalDegree() == 3 and self.a2.GetTotalNumHs() == 0:
            # detect tertiary sulfonamide N
            # root N nbr has to be
            oxy_count = 0
            c_nbrs_of_s = 0
            c_nbrs_of_n = 0
            for a2_nbr in nbrs_a2:
                if a2_nbr.GetAtomicNum() == 6:
                    c_nbrs_of_n += 1
                elif a2_nbr.GetAtomicNum() == 16 and a2_nbr.GetDegree() == 4:
                    for s_nbr in a2_nbr.GetNeighbors():
                        if s_nbr.GetIdx() == self.a2.GetIdx():  # skip root N
                            continue
                        elif s_nbr.GetAtomicNum() == 8:
                            if self.mol_2.GetBondBetweenAtoms(s_nbr.GetIdx(),
                                                              a2_nbr.GetIdx()).GetBondTypeAsDouble() == 2.0:
                                oxy_count += 1
                        elif s_nbr.GetAtomicNum() == 6:
                            c_nbrs_of_s += 1
                        else:
                            return False
                else:
                    return False
            # there must be exactly 1 C nbr of the sulfonamied N (excluding a1 itself) because N must be tertiary
            if oxy_count == 2 and c_nbrs_of_s == 1 and c_nbrs_of_n == 1:
                return True

            return False
        return False

    # Piperidine-Indole
    def piperidine_indole(self):
        # aroN-olefinC
        if self.a1.GetIsAromatic() and self.a1.IsInRingSize(5) and self.a1.GetDegree() == 3 \
                and not self.a2.GetIsAromatic() and self.a2.IsInRingSize(6) and self.a2.GetDegree() == 3:
            if self._match_indole() and self._match_monounsaturated_piperidine():
                return True

            return False
        return False

    # Piperidine-Indole
    def _match_monounsaturated_piperidine(self):
        """self.a2 - root atom of the monounsaturated piperide ring
            self.mol_2 - the corresponding mol object"""

        # loop over first order of neighbors of the root C atom in the 6-ring
        db_count = 0
        n_count = 0
        c_nbrs_1st_order = []
        for a in self.a2.GetNeighbors():
            if a.GetAtomicNum() == 0:  # skip dummy atom in case of connecting
                continue
            elif a.GetAtomicNum() == 7:  # only 1 N neighbor allowed in case of cutting
                n_count += 1
            elif a.GetAtomicNum() == 6:  # exactly 2 C neighbors allowed
                if a.GetDegree() != 2 or a.GetIsAromatic():
                    continue
                c_nbrs_1st_order.append(a)
                if self.mol_2.GetBondBetweenAtoms(a.GetIdx(), self.a2.GetIdx()).GetBondTypeAsDouble() == 2:
                    db_count += 1
            else:  # no other heteroatom nbrs allowed
                return False
        if db_count != 1 or n_count > 1 or len(c_nbrs_1st_order) != 2:
            return False

        # loop over first order of neighbors (secondary carbons - 1 aliphatic, 1 olefinic) in the 6-ring in ortho position to the root C atom
        c_nbrs_2nd_order = []
        for a1 in c_nbrs_1st_order:
            for a in a1.GetNeighbors():
                if a.GetIdx() == self.a2.GetIdx():  # skip the query C atom a neighbor
                    continue
                if a.GetAtomicNum() != 6 or a.GetDegree() != 2 or \
                        a.GetTotalNumHs() != 2:  # must be secondary aliphatic carbons
                    return False
                c_nbrs_2nd_order.append(a)

        # loop over second order of neighbors (secondary aliphatic carbons) in the 6-ring in meta position to the root C atom
        para_n = []
        for a2 in c_nbrs_2nd_order:
            for a in a2.GetNeighbors():
                if a.GetIdx() in [x.GetIdx() for x in
                                  c_nbrs_1st_order]:  # skip the 1st order neighbors of the root C atom
                    continue
                if a.GetAtomicNum() != 7 or a.GetDegree() not in (
                        2, 3):  # only secondary or tertiary N in para position to the root C atom is allowed
                    return False
                para_n.append(a)
        if len(set([x.GetIdx() for x in
                    para_n])) == 1:  # if this is the same para N atom for both 2nd order C nbrs of the root C atom
            return True  # there is a monounsaturated piperide ring

    def _match_indole(self):
        """self.a1 - root atom (aromatic N) of the indole ring
            self.mol_1 - the corresponding mol object"""

        # loop over the neighbors of the root aromatic N atom (C2 and C9)
        # C9 can be substituted, C1 is tertiary aromatic
        first_order_nbrs = []
        for a in self.a1.GetNeighbors():
            if not a.GetIsAromatic():  # skip the root C atom of the putative piperidine ring in case of cutting or the R-atom (cannot be aromatic) in case of connecting
                continue
            if a.GetAtomicNum() != 6 or not a.GetIsAromatic():  # indole has only 1 N as heteroatom, all other atoms are aromatic Cs
                return False
            first_order_nbrs.append(a)
        # only C2 is tertiary aromatic (simultaneously in 5- and 6-ring); C9 can be only in the 5-ring
        if sum([x.IsInRingSize(6) for x in first_order_nbrs]) != 1:
            return False

        # loop over second order neighbors of the root aromatic N atom
        C2_nbrs = []
        C9_nbrs = []
        for a1 in first_order_nbrs:
            for a in a1.GetNeighbors():
                if a.GetIdx() == self.a1.GetIdx():  # skip root N atom
                    continue
                if a.GetAtomicNum() != 6 or not a.GetIsAromatic():
                    continue
                if a1.IsInRingSize(6):  # nbrs of C2
                    if not a.IsInRingSize(6):  # all must be part of the 6-ring
                        return False
                    C2_nbrs.append(a)
                else:  # nbrs of C9
                    # add an aromatic C from a 5-ring system only if it is a secondary C i.e. is unsubstituted
                    # thus, potential aromatic Cs from other 5-ring systems that may act as substituents at C9
                    # are excluded because they would always be tartiary aromatic Cs
                    if a.IsInRingSize(5) and a.GetDegree() == 2:
                        C9_nbrs.append(a)
                        # only C8 should be detected
        if len(C9_nbrs) != 1:
            return False
            # C7 is tertiary aromatic; C3 must secondary aromatic i.e. unsubstituted
        if sum([x.GetTotalNumHs() for x in C2_nbrs]) != 1:
            return False

        second_order_nbrs = C2_nbrs + C9_nbrs

        # loop over third-order neighbors of the root N atom - C4, C6, C7 and skip already visited atoms
        seen_atoms = first_order_nbrs + second_order_nbrs
        third_order_nbrs = []
        for a2 in second_order_nbrs:
            for a in a2.GetNeighbors():
                if a.GetIdx() in [x.GetIdx() for x in seen_atoms]:  # skip seen atoms
                    continue
                if a.GetAtomicNum() != 6 or not a.GetIsAromatic() or not a.IsInRingSize(6):
                    return False
                if a2.GetDegree() == 3:  # nbrs of C7
                    if a.GetTotalNumHs() == 1:  # C6 must be unsubstituted
                        third_order_nbrs.append(a)
                else:  # nbrs of C3
                    third_order_nbrs.append(a)  # C4 may be substituted or unsubstituted
        # both C4 and C6 must be detected
        if len(third_order_nbrs) != 2:
            return False

        # loop over fourth-order neighbors of the root N atom after skipping already seen atoms
        seen_atoms += third_order_nbrs
        fourth_order_nbrs = []
        for a3 in third_order_nbrs:
            # if C4 is substituted (tertiary aromatic), skip iterating over its neighbors
            # and move on to the nbrs of C6 where C5 will be automatically detected as 4th-order nbr
            if a3.GetDegree() != 2:
                continue
            for a in a3.GetNeighbors():
                if a.GetIdx() in [x.GetIdx() for x in seen_atoms]:
                    continue
                if a.GetAtomicNum() != 6 or not a.GetIsAromatic() or not a.IsInRingSize(6):
                    continue
                fourth_order_nbrs.append(a)
        # there should be only C5 (can be substituted)
        if len(set([x.GetIdx() for x in fourth_order_nbrs])) == 1:
            return True