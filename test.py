import numpy as np
from warnings import warn
from biotite.structure import AtomArrayStack
from biotite.structure.io import pdb
from biotite.structure.io.pdb.hybrid36 import encode_hybrid36, max_hybrid36_number
from biotite.structure.box import unitcell_from_vectors
import pdbhelper


def set_structure(array, hybrid36=False):
    """
    Test PDBFile.set_structure speed-ups
    
    Parameters
    ----------
    array : AtomArrayStack
        The array or stack to be saved into this file. If a stack
        is given, each array in the stack is saved as separate
        model.
    hybrid36: bool, optional
        Defines wether the file should be written in hybrid-36
        format.
    
    Notes
    -----
    If `array` has an associated :class:`BondList`, ``CONECT``
    records are also written for all non-water hetero residues
    and all inter-residue connections.
    """
    natoms = array.array_length()
    annot_categories = array.get_annotation_categories()
    record = np.char.array(np.where(array.hetero, "HETATM", "ATOM"))
    # Check for optional annotation categories
    if "atom_id" in annot_categories:
        atom_id = array.atom_id
    else:
        atom_id = np.arange(1, natoms + 1)
    if "b_factor" in annot_categories:
        b_factor = np.char.array([f"{b:>6.2f}" for b in array.b_factor])
    else:
        b_factor = np.char.array(np.full(natoms, "  0.00", dtype="U6"))
    if "occupancy" in annot_categories:
        occupancy = np.char.array([f"{o:>6.2f}" for o in array.occupancy])
    else:
        occupancy = np.char.array(np.full(natoms, "  1.00", dtype="U6"))
    if "charge" in annot_categories:
        charge = np.char.array(
            [str(np.abs(charge)) + "+" if charge > 0 else
                (str(np.abs(charge)) + "-" if charge < 0 else "")
                for charge in array.get_annotation("charge")]
        )
    else:
        charge = np.char.array(np.full(natoms, "  ", dtype="U2"))

    # Do checks on atom array (stack)
    if hybrid36:
        max_atoms = max_hybrid36_number(5)
        max_residues = max_hybrid36_number(4)
    else:
        max_atoms, max_residues = 99999, 9999
    if array.array_length() > max_atoms:
        warn(f"More then {max_atoms:,} atoms per model")
    if (array.res_id > max_residues).any():
        warn(f"Residue IDs exceed {max_residues:,}")
    if np.isnan(array.coord).any():
        raise ValueError("Coordinates contain 'NaN' values")
    if any([len(name) > 1 for name in array.chain_id]):
        raise ValueError("Some chain IDs exceed 1 character")
    if any([len(name) > 3 for name in array.res_name]):
        raise ValueError("Some residue names exceed 3 characters")
    if any([len(name) > 4 for name in array.atom_name]):
        raise ValueError("Some atom names exceed 4 characters")

    if hybrid36:
        pdb_atom_id = np.char.array(
            [encode_hybrid36(i, 5) for i in atom_id]
        )
        pdb_res_id = np.char.array(
            [encode_hybrid36(i, 4) for i in array.res_id]
        )
    else:
        # Atom IDs are supported up to 99999,
        # but negative IDs are also possible
        pdb_atom_id = np.char.array(np.where(
            atom_id > 0,
            ((atom_id - 1) % 99999) + 1,
            atom_id
        ).astype(str))
        # Residue IDs are supported up to 9999,
        # but negative IDs are also possible
        pdb_res_id = np.char.array(np.where(
            array.res_id > 0,
            ((array.res_id - 1) % 9999) + 1,
            array.res_id
        ).astype(str))
    
    names = np.char.array(
        [f" {atm}" if len(elem) == 1 and len(atm) < 4 else atm
            for atm, elem in zip(array.atom_name, array.element)]
    )
    res_names = np.char.array(array.res_name)
    chain_ids = np.char.array(array.chain_id)
    ins_codes = np.char.array(array.ins_code)
    spaces = np.char.array(np.full(natoms, " ", dtype="U1"))
    elements = np.char.array(array.element)

    first_half = (
        record.ljust(6) +
        pdb_atom_id.rjust(5) +
        spaces +
        names.ljust(4) +
        spaces + res_names.rjust(3) + spaces + chain_ids +
        pdb_res_id.rjust(4) + ins_codes.rjust(1)
    )

    second_half = (
        occupancy + b_factor + 10 * spaces +
        elements.rjust(2) + charge.rjust(2)
    )

    coords = array.coord
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]
    
    lines = []
    # Prepend a single CRYST1 record if we have box information
    if array.box is not None:
        box = array.box
        if len(box.shape) == 3:
            box = box[0]
        a, b, c, alpha, beta, gamma = unitcell_from_vectors(box)
        lines.append(
            f"CRYST1{a:>9.3f}{b:>9.3f}{c:>9.3f}"
            f"{np.rad2deg(alpha):>7.2f}{np.rad2deg(beta):>7.2f}"
            f"{np.rad2deg(gamma):>7.2f} P 1           1"
        )
    is_stack = coords.shape[0] > 1
    for model_num, coord_i in enumerate(coords, start=1):
        # for an ArrayStack, this is run once
        # only add model lines if is_stack
        if is_stack:
            lines.append(f"MODEL     {model_num:4}")
        # Bundle non-coordinate data to simplify iteration
        lines.extend(
            [f"{start:27}   {x:>8.3f}{y:>8.3f}{z:>8.3f}{end:26}"
                for start, (x, y, z), end in
                zip(first_half, coord_i, second_half)]
        )
        if is_stack:
            lines.append("ENDMDL")
    return lines


def set_structure_cython(array, hybrid36=False):
    """
    Test PDBFile.set_structure speed-ups
    
    Parameters
    ----------
    array : AtomArrayStack
        The array or stack to be saved into this file. If a stack
        is given, each array in the stack is saved as separate
        model.
    hybrid36: bool, optional
        Defines wether the file should be written in hybrid-36
        format.
    
    Notes
    -----
    If `array` has an associated :class:`BondList`, ``CONECT``
    records are also written for all non-water hetero residues
    and all inter-residue connections.
    """
    natoms = array.array_length()
    annot_categories = array.get_annotation_categories()
    is_hetero = np.where(array.hetero, 1, 0).astype(np.intc)
    # Check for optional annotation categories
    if "atom_id" in annot_categories:
        atom_id = array.atom_id
    else:
        atom_id = np.arange(1, natoms + 1)
    if "b_factor" in annot_categories:
        b_factor = array.b_factor
    else:
        b_factor = np.zeros(natoms, dtype=np.float32)
    if "occupancy" in annot_categories:
        occupancy = array.occupancy
    else:
        occupancy = np.ones(natoms, dtype=np.float32)
    if "charge" in annot_categories:
        charge = np.array(
            [str(np.abs(charge)) + "+" if charge > 0 else
             (str(np.abs(charge)) + "-" if charge < 0 else "")
             for charge in array.get_annotation("charge")],
            dtype=np.bytes_
        )
    else:
        charge = np.full(natoms, "  ", dtype=np.bytes_)

    # Do checks on atom array (stack)
    if hybrid36:
        max_atoms = max_hybrid36_number(5)
        max_residues = max_hybrid36_number(4)
    else:
        max_atoms, max_residues = 99999, 9999
    if array.array_length() > max_atoms:
        warn(f"More then {max_atoms:,} atoms per model")
    if (array.res_id > max_residues).any():
        warn(f"Residue IDs exceed {max_residues:,}")
    if np.isnan(array.coord).any():
        raise ValueError("Coordinates contain 'NaN' values")
    if any([len(name) > 1 for name in array.chain_id]):
        raise ValueError("Some chain IDs exceed 1 character")
    if any([len(name) > 3 for name in array.res_name]):
        raise ValueError("Some residue names exceed 3 characters")
    if any([len(name) > 4 for name in array.atom_name]):
        raise ValueError("Some atom names exceed 4 characters")

    if hybrid36:
        pdb_atom_id = np.array(
            [encode_hybrid36(i, 5) for i in atom_id], dtype=np.bytes_
        )
        pdb_res_id = np.array(
            [encode_hybrid36(i, 4) for i in array.res_id], dtype=np.bytes_
        )
    else:
        # Atom IDs are supported up to 99999,
        # but negative IDs are also possible
        pdb_atom_id = np.where(
            atom_id > 0,
            ((atom_id - 1) % 99999) + 1,
            atom_id
        ).astype(np.bytes_)
        # Residue IDs are supported up to 9999,
        # but negative IDs are also possible
        pdb_res_id = np.where(
            array.res_id > 0,
            ((array.res_id - 1) % 9999) + 1,
            array.res_id
        ).astype(np.bytes_)
    
    names = array.atom_name.astype(np.bytes_)
    res_names = array.res_name.astype(np.bytes_)
    chain_ids = array.chain_id.astype(np.bytes_)
    ins_codes = array.ins_code.astype(np.bytes_)
    elements = array.element.astype(np.bytes_)

    first_half, second_half = pdbhelper.noncoordinate_pdb_data(
        is_hetero, pdb_atom_id, names, res_names, chain_ids, pdb_res_id,
        ins_codes, occupancy, b_factor, elements, charge
    )

    coords = array.coord
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]
    
    lines = []
    # Prepend a single CRYST1 record if we have box information
    if array.box is not None:
        box = array.box
        if len(box.shape) == 3:
            box = box[0]
        a, b, c, alpha, beta, gamma = unitcell_from_vectors(box)
        lines.append(
            f"CRYST1{a:>9.3f}{b:>9.3f}{c:>9.3f}"
            f"{np.rad2deg(alpha):>7.2f}{np.rad2deg(beta):>7.2f}"
            f"{np.rad2deg(gamma):>7.2f} P 1           1"
        )
    is_stack = coords.shape[0] > 1
    for model_num, coord_i in enumerate(coords, start=1):
        # for an ArrayStack, this is run once
        # only add model lines if is_stack
        if is_stack:
            lines.append(f"MODEL     {model_num:4}")
        # Bundle non-coordinate data to simplify iteration
        lines.extend(pdbhelper.coordinate_record_lines(
            first_half, coord_i, second_half
        ))
        if is_stack:
            lines.append("ENDMDL")
    return lines


if __name__ == "__main__":
    import timeit
    
    N = 100

    pdb_file = pdb.PDBFile.read("1GYA.pdb")
    atoms = pdb_file.get_structure()

    time = timeit.timeit(
        "set_structure(atoms)",
        "from __main__ import set_structure, atoms",
        number=N
    )
    print(f"numpy: {time * 1e3 / N :.2f} ms")

    time = timeit.timeit(
        "set_structure_cython(atoms)",
        "from __main__ import set_structure_cython, atoms",
        number=N
    )
    print(f"cython: {time * 1e3 / N :.2f} ms")
