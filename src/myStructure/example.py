import os

from aiida import orm, load_profile
from aiida.engine import submit
from .structure import myStructureData
from pymatgen.core import Structure

from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_quantumespresso.common.types import SpinType
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
def main():
    """Example of StructureData with magmoms"""
    load_profile()
    file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        # "structures/Fe_afm_bcc.mcif"
        "structures/Fe_bcc.mcif"
    )
    smag1 = Structure.from_file(file, primitive=True)
    structure = myStructureData(pymatgen=smag1)
    print(structure.base.attributes.all)
    for kind in structure.mykinds:
        print(kind.get_magmom_coord())
    builder = PwBandsWorkChain.get_builder_from_protocol(
        code="pw_dev@slurm",
        structure=structure,
        protocol="moderate",
        spin_type=SpinType.NON_COLLINEAR,
    )

    builder.structure = structure.to_aiida_structure()
    builder.pop('relax')
    parallelization = {
        "num_mpiprocs_per_machine": 12,
        "npool": 3,
        "max_wallclock_seconds": 24 * 3600
    }
    set_parallelization(builder, parallelization, process_class=PwBandsWorkChain)
    print_builder(builder)
    # submit(builder)

if __name__ == "__main__":
    main()