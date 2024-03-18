"""Module that registers the main classes and functions of the ATGraFS-ext."""

import logging

import numpy as np
import itertools as it

from collections import defaultdict

from atgrafsE.utils.topology import read_topologies_database
from atgrafsE.utils.topology import Topology
from atgrafsE.utils.sbu import read_sbu_database

from atgrafsE.framework import Framework

from atgrafsE.utils.sbu import SBU

logger = logging.getLogger(__name__)


class Autografs:
    """Framework maker class to generate ASE Atoms objects from topologies."""

    def __init__(self, topology_path=None, sbu_path=None, use_defaults=True, update=False):
        """Initialize the class by reading the database."""
        logger.info("{0:*^80}".format("*"))
        logger.info("* {0:^76} *".format("AuToGraFS"))
        logger.info("* {0:^76} *".format(
            "Automatic Topological Generator for Framework Structures Extended"
        ))
        logger.info("{0:*^80}".format("*"))
        logger.info("")
        logger.info("Reading the topology database.")
        if topology_path is not None:
            use_defaults = False

        self.topologies = read_topologies_database(
            path=topology_path,
            use_defaults=use_defaults,
            update=update
        )

        logger.info("Reading the building units database.")
        self.sbu = read_sbu_database(
            path=sbu_path
        )

        # container for current topology
        self.topology = None
        # container for current sbu mapping
        self.sbu_dict = {}
        # Used Slot
        self.used_slots = []
        # dict of sbu per slots
        self.sbu_per_slots = {}
        # probabilities per sbu in case that them find in the same slot
        self.prob_sbu = defaultdict(dict)
        self.prob_per_slots = {}
        self.type_per_slots = {}
        logger.info("")

    def _entry_format(self, entry):
        """Format entries."""
        # Revisa el tipo de entrada y la adapta.
        if isinstance(entry, list):
            entry_formated = []
            for elements in entry:
                if isinstance(elements, str):
                    entry_formated.append((elements, 1.0))
                elif isinstance(elements, tuple):
                    assert len(elements) == 2, "You have to use only two values if it's a tuple"
                    name, p = elements
                    name = str(name)
                    p = float(p)
                    entry_formated.append((name, p))
            entry = entry_formated.copy()
        else:
            if isinstance(entry, str):
                # 1) la entrada es un str ---> list[(str, float)]
                entry = [(entry, 1.0)]

            elif isinstance(entry, tuple):
                # 2) la entrada es una tuple ---> list[(str, float)]
                assert len(entry) == 2, "You have to use only two values if it's a tuple"
                name, p = entry
                name = str(name)
                p = float(p)
                entry = [(name, p)]

        return entry

    def set_topology(self, topology_name, supercell=(1, 1, 1)):
        """Create and store the topology object."""
        logger.info("Topology set to --> {topo}".format(topo=topology_name.upper()))
        # make the supercell prior to alignment
        if isinstance(supercell, int):
            supercell = (supercell, supercell, supercell)

        topology_atoms = self.topologies[topology_name]
        if supercell != (1, 1, 1):
            logger.info("{0}x{1}x{2} supercell of the topology is used.".format(*supercell))
            topology_atoms *= supercell

        # make the Topology object
        logger.info("Analysis of the topology.")
        topology = Topology(name=topology_name, atoms=topology_atoms)

        self.topology = topology

    def _checking_prob_corr(self, choice_sbu=False):
        """
        Check the defined probabilities and even correct them.

        At the moment it will not be allowed for a ligand to compete with a metal in the same slot
        for, if this is found the probability for the ligand will be set to 0.0.
        """
        for slot in self.type_per_slots:
            types_in = set(self.type_per_slots[slot])
            if len(types_in) != 1:
                for i, tp in enumerate(self.type_per_slots[slot]):
                    if not choice_sbu:
                        if tp == "ligand":
                            self.prob_per_slots[slot][i] = 0.0
                    else:
                        if tp == "metal":
                            self.prob_per_slots[slot][i] = 0.0

            prob = np.array(self.prob_per_slots[slot])
            prob /= np.sum(prob)
            self.prob_per_slots[slot] = list(prob)

    def _slots_per_sbu(self, sbu, coercion):
        """Find the compatible slots for an SBU."""
        # get compatible slot from topology
        slots = self.topology.has_compatible_slots(sbu=sbu, coercion=coercion)
        for index, shape in self.topology.shapes.items():
            # get slots from topolgy
            # here, should accept weights also
            shape = tuple(shape)

            if shape not in slots:
                logger.debug("Unfilled slot at index {idx}".format(idx=index))
                continue

            logger.debug("Slot {sl}: {sb} founded.".format(sl=index, sb=sbu.name))
            self.sbu_per_slots[index].append(sbu)
            self.prob_per_slots[index].append(self.prob_sbu[sbu.name]["p"])
            self.type_per_slots[index].append(self.prob_sbu[sbu.name]["type"])

    def _build_dict_SBU_selected(self, list_SBUs, coercion, choice_sbu):
        """
        Build a dictionary of selected SBUs.

        Where the slots will be the keys and the SBUs the values. The selection of a given SBU
        will depend on a given probability.
        """
        filled = True
        for sbu in list_SBUs:
            self._slots_per_sbu(sbu, coercion)

        # Corrects the probabilities
        self._checking_prob_corr(choice_sbu)

        # It checks if all slots are full and selects the SBU according to its probabilities.
        for slot in self.sbu_per_slots:
            if len(self.sbu_per_slots[slot]) == 0:
                logger.info("\t ---> Slot {} empty".format(slot))
                filled = False
                continue
            sbu_chosen = np.random.choice(
                self.sbu_per_slots[slot], p=self.prob_per_slots[slot]
            )

            self.sbu_dict[slot] = sbu_chosen.copy()
            self.used_slots.append(slot)

        return filled

    def make_MOF(self, topology_name, metal=None, sbu_selected=None, supercell=(1, 1, 1,), coercion=False, edge_bonds=False, choice_sbu=False):
        """
        Return of the best mof found.

        This will be the new core method of AuToGraFS.

        This is a test mode that takes the name of a metal or an sbu (also a list of these), and
        these will be positioned on the nodes of the topology. The EDGES will be taken as links
        between the added compounds.

        It is possible to add the insertion probability as a tuple: (name_sbu, 0.5).

        If a metal is detected, just indicate the element and it will be assigned a geometry
        detected in the topology.

        The links will be built from an internal SBU consisting of two X-X atoms with a distance of
        1.5 angs.

        Parameters:
        -----------
        topology_name : str, list of str, optional

        metal : str, or list of str, or tuple, or list of tuples

        sbu : str, or list of str, or tuple, or list of tuples

        supercell : tuple(x3) of int
            (optional) Creates a supercell pre-treatment.

        coercion : bool
            (optional) Force the compatibility to only consider the multiplicity of SBU.
        """
        logger.info("{0:-^80}".format(" Making MOF "))
        logger.info("")

        # only set the topology if not already done
        if topology_name is not None:
            self.set_topology(topology_name=topology_name, supercell=supercell)

        assert self.topology is not None, "You must select a topology to search for slots."

        if not self.topology.analyze_complet:
            logger.info("Error in symmetry assignment")
            logger.info("{0:-^80}".format(" MOF not generated "))
            return None

        pg_list = set([self.topology.pointgroups[key] for key in self.topology.pointgroups])
        logger.info("Points group in topology: {}".format(" ".join(pg_list)))

        logger.info("N={} shapes in topology".format(len(self.topology.shapes.items())))
        self.slots_available = [items[0] for items in self.topology.shapes.items()]
        logger.debug("Slots available: {}".format(" ".join([str(i) for i in self.slots_available])))

        for index in self.slots_available:
            self.sbu_per_slots[index] = []
            self.prob_per_slots[index] = []
            self.type_per_slots[index] = []

        if sbu_selected is None and metal is None:
            raise ValueError("No fragment has been defined for metal or sbu.")

        # We start by studying the specified metal center.
        if metal is not None:
            logger.info("Metal center defined:")
            metal = self._entry_format(metal)

            for name, p in metal:
                logger.info("\tmetal: {} ({})".format(name, p))
        else:
            metal = []

        if sbu_selected is not None:
            logger.info("SBUs defined:")
            sbu_selected = self._entry_format(sbu_selected)

            for name, p in sbu_selected:
                logger.info("\tsbu: {} ({})".format(name, p))
        else:
            sbu_selected = []

        # A list of SBUs will be created before searching for available slots.
        list_SBUs = []

        # If the option to take the edges as bonds is activated, a theoretical X--X molecule will
        # be created.
        if edge_bonds:
            # Adding model for D*h slots
            name = "X--X"
            sbu = SBU(name=name)
            sbu.define_geometry("", "D*h")
            logger.info("Name: {} - Point group: {}".format(name, sbu.pg))
            list_SBUs.append(sbu)
            self.prob_sbu[name]["p"] = 1.0
            self.prob_sbu[name]["type"] = "ligand"

        # First the metals will be added to the list.
        for pg in pg_list:
            # At the moment the D*h option is not configured for metals.
            if pg == "D*h":
                continue

            for element, p in metal:
                name = f"{element}_{pg}"
                try:
                    sbu = SBU(name=name)
                    sbu.define_geometry(element, pg)
                    list_SBUs.append(sbu.copy())
                    self.prob_sbu[name]["p"] = p
                    self.prob_sbu[name]["type"] = "metal"
                    logger.info("Name: {} - Point group: {}".format(name, sbu.pg))
                except KeyError:
                    logger.info("No geometry {} was found in the database.".format(pg))
                    break

        # The selected SBUs are now added
        for name, p in sbu_selected:
            list_SBUs.append(SBU(name=name, atoms=self.sbu[name]))
            self.prob_sbu[name]["p"] = p
            self.prob_sbu[name]["type"] = "ligand"

        # All slots have a candidate
        filled_slots = self._build_dict_SBU_selected(list_SBUs, coercion, choice_sbu)
        if not filled_slots:
            logger.info("Not all slots were filled.")
            logger.info("{0:-^80}".format(" MOF not generated "))
            return None
        else:
            logger.info("All slots have at least one candidate")

        # identify the corresponding SBU
        logger.info("Scheduling the SBU to slot alignment.")

        # some logging
        self.log_sbu_dict()

        # MOF is initialized.
        aligned = Framework()
        aligned.set_topology(self.topology)

        # Alignment start
        aligned.alignment_sbu(self.sbu_dict)
        if not aligned.complete_alignment:
            logger.info("Alignment has not been completed")
            logger.info("{0:-^80}".format(" MOF not generated "))
            return None
        else:
            logger.info("Complete alignment")

        logger.info("{0:-^80}".format(" Finished MOF generation "))
        logger.info("")
        logger.info("{0:-^80}".format(" Post-treatment "))
        logger.info("")

        return aligned

    def log_sbu_dict(self):
        """Do some logging on the chosen SBU mapping."""
        logger.info("Summary of slots:")
        for idx in self.sbu_dict:
            logger.info("\tSlot {}".format(idx))
            sbu = self.sbu_dict[idx]
            logger.info("\t   |--> SBU: {sbn}".format(sbn=sbu.name))
            logger.info("\t   |--> PG: {pg}".format(pg=sbu.pg))

    def get_topology(self, topology_name=None):
        """Generate and return a Topology object.

        Parameters:
        -----------
        topology_name : str
            The name of the selected topology.
        """
        topology_atoms = self.topologies[topology_name]
        return Topology(name=topology_name, atoms=topology_atoms)

    def list_available_frameworks(self, topology_name=None, from_list=[], coercion=False):
        """Return a list of sbu_dict covering all the database.

        Parameters
        ----------
        topology_name : str
            Name of the topology to use.

        from_list : list of str
            Only consider SBU from this list.

        coercion : bool
            Wether to force compatibility by coordination alone.
        """
        av_sbu = self.list_available_sbu(
            topology_name=topology_name,
            from_list=from_list,
            coercion=coercion
        )

        dicts = []
        for product in it.product(*av_sbu.values()):
            tmp_d = {}
            for k, v in zip(av_sbu.keys(), product):
                tmp_d.update({kk: v for kk in k})
            dicts.append(tmp_d)

        return dicts

    def list_available_sbu(self, topology_name=None, from_list=[], coercion=False):
        """Return the dictionary of compatible SBU.

        Filters the existing SBU by shape until only those compatible with a slot within the
        topology are left.

        Parameters:
        -----------
        topology_name : str
            Name of the topology in the database.

        from_list : list of str
            Only consider SBU from this list.

        coercion : bool
            Wether to force compatibility by coordination alone.
        """
        av_sbu = defaultdict(list)
        if from_list:
            sbu_names = from_list
        else:
            sbu_names = list( self.sbu.keys())
        sbu_names = sorted(sbu_names)
        if topology_name is not None or self.topology is not None:
            if topology_name is not None:
                topology = Topology(name=topology_name,
                                    atoms=self.topologies[topology_name])
            else:
                topology = self.topology
            logger.info("List of compatible SBU with topology {t}:".format(t=topology.name))
            sbu_list = []
            logger.info("\tShape analysis of {} available SBU...".format(len(self.sbu)))
            for sbuk in sbu_names:
                sbuv = self.sbu[sbuk]
                try:
                    sbu = SBU(name=sbuk,
                              atoms=sbuv)
                except Exception as exc:
                    logger.debug("SBU {k} not loaded: {exc}".format(k=sbuk,exc=exc))
                    continue
                sbu_list.append(sbu)
            for sites in topology.equivalent_sites:
                logger.info("\tSites considered : {}".format(", ".join([str(s) for s in sites])))
                shape = topology.shapes[sites[0]]
                for sbu in sbu_list:
                    is_compatible = sbu.is_compatible(shape, coercion=coercion)
                    if is_compatible:
                        logger.info("\t\t|--> {k}".format(k=sbu.name))
                        av_sbu[tuple(sites)].append(sbu.name)
            return dict(av_sbu)
        else:
            logger.info("Listing full database of SBU.")
            av_sbu = list(self.sbu.keys())
            av_sbu = sorted(av_sbu)
            return av_sbu

    def list_available_topologies(self, sbu_names=[], full=True, max_size=100, from_list=[], pbc="all", coercion=False):
        """Return a list of topologies compatible with the SBUs.

        For each sbu in the list given in input, refines first by coordination then by shapes
        within the topology. Thus, we do not need to analyze every topology.

        Parameters:
        -----------
        sbu_names : list of str
            List of sbu names.

        full : bool
            Wether the topology is entirely represented by the sbu.

        max_size : int
            Maximum size of in SBU numbers of topologies to consider.

        from_list : list of str
            Only consider topologies from this list.

        pbc: str
            Defines the periodicity: all, 2D, 3D.

        coercion : bool
            Wether to force compatibility by coordination alone.
        """
        these_topologies_names = self.topologies.keys()
        if max_size is None:
            max_size = 999999
        if from_list:
            these_topologies_names = from_list
        if pbc == "2D":
            logger.info("only considering 2D periodic topologies.")
            these_topologies_names = [tk for tk, tv in self.topologies.items() if sum(tv.pbc) == 2]
        elif pbc == "3D":
            logger.info("only considering 3D periodic topologies.")
            these_topologies_names = [tk for tk, tv in self.topologies.items() if sum(tv.pbc) == 3]
        elif pbc != "all":
            logger.info("pbc keyword has to be '2D','3D' or 'all'. Assumed 'all'.")
        these_topologies_names = sorted(these_topologies_names)
        if sbu_names:
            logger.info("Checking topology compatibility.")
            topologies = []
            sbu = [SBU(name=n, atoms=self.sbu[n]) for n in sbu_names]
            for tk in these_topologies_names:
                tv = self.topologies[tk]
                if max_size is None or len(tv) > max_size:
                    logger.debug("\tTopology {tk} to big : size = {s}.".format(tk=tk, s=len(tv)))
                    continue
                try:
                    topology = Topology(name=tk, atoms=tv)
                except Exception as exc:
                    logger.debug("Topology {tk} not loaded: {exc}".format(tk=tk, exc=exc))
                    continue
                filled = {shape: False for shape in topology.get_unique_shapes()}
                slots_full = [topology.has_compatible_slots(s, coercion=coercion) for s in sbu]
                for slots in slots_full:
                    for slot in slots:
                        filled[slot] = True
                if all(filled.values()):
                    logger.info("\tTopology {tk} fully available.".format(tk=tk))
                    topologies.append(tk)
                elif any(filled.values()) and not full:
                    logger.info("\tTopology {tk} partially available.".format(tk=tk))
                    topologies.append(tk)
                else:
                    logger.debug("\tTopology {tk} not available.".format(tk=tk))
        else:
            logger.info("Listing full database of topologies.")
            topologies = list(self.topologies.keys())
            topologies = sorted(topologies)
        return topologies
