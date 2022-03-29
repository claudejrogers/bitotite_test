cimport cython
cimport numpy as np
import numpy as np

from libc.stdio cimport sprintf
from libc.string cimport memset, strcat, strlen, strncpy

np.import_array()

DEF START_SIZE = 31
DEF END_SIZE = 27

S1 = np.dtype("S1")
S2 = np.dtype("S2")
S3 = np.dtype("S3")
S4 = np.dtype("S4")
S5 = np.dtype("S5")
S27 = np.dtype("S27")
S31 = np.dtype("S31")


cdef bytes _sprintf_coord_start(
    int is_hetero,
    char *atom_id,
    char *atom,
    char *resname,
    char *chain,
    char *res_id,
    char *icode,
    char *element
):
    cdef char buffer[START_SIZE]
    cdef char atom_name[5]
    memset(buffer, 0, START_SIZE)
    memset(atom_name, 0, 5)
    if strlen(atom) < 4 and strlen(element) == 1:
        strncpy(atom_name, " ", 1)
        strcat(atom_name, atom)
    else:
        strncpy(atom_name, atom, 4)
    sprintf(
        buffer,
        "%-6s%5s %-4s %3s %1s%4s%1s   ",
        "HETATM" if is_hetero else "ATOM",
        atom_id,
        atom_name,
        resname,
        chain,
        res_id,
        icode
    )
    return buffer  # [:START_SIZE].decode('UTF-8')


cdef bytes _sprintf_coord_end(
    float occupancy,
    float temp_factor,
    char *element,
    char *charge
):
    cdef char buffer[END_SIZE]
    memset(buffer, 0, END_SIZE)
    sprintf(
        buffer,
        "%6.2f%6.2f          %2s%2s",
        occupancy,
        temp_factor,
        element,
        charge
    )
    return buffer # [:END_SIZE].decode('UTF-8')


cdef unicode _sprintf_coord_line(
    char *start,
    float x,
    float y,
    float z,
    char *end
):
    cdef char buffer[82]
    cdef char cbuffer[25]
    memset(buffer, 0, 82)
    memset(cbuffer, 0, 25)
    sprintf(cbuffer, "%8.3f%8.3f%8.3f", x, y, z)
    strncpy(buffer, start, START_SIZE)
    strcat(buffer, cbuffer)
    strcat(buffer, end)
    return buffer[:82].decode('UTF-8')


@cython.boundscheck(False)
@cython.wraparound(False)
def noncoordinate_pdb_data(
    int[:] is_hetero,
    np.ndarray atom_id,
    np.ndarray atom,
    np.ndarray resname,
    np.ndarray chain,
    np.ndarray res_id,
    np.ndarray icode,
    float[:] occupancy,
    float[:] temp_factor,
    np.ndarray element,
    np.ndarray charge
):
    cdef size_t natoms = is_hetero.shape[0]
    cdef size_t i
    cdef np.ndarray start = np.empty([natoms], dtype=S31)
    cdef np.ndarray end = np.empty([natoms], dtype=S27)
    # assert inputs
    atom_id = atom_id.astype(S5)
    atom = atom.astype(S4)
    resname = resname.astype(S3)
    chain = chain.astype(S1)
    res_id = res_id.astype(S4)
    icode = icode.astype(S1)
    element = element.astype(S2)
    charge = charge.astype(S2)
    for i in range(natoms):
        start[i] = _sprintf_coord_start(
            is_hetero[i], atom_id[i], atom[i], resname[i], chain[i],
            res_id[i], icode[i], element[i]
        )
        end[i] = _sprintf_coord_end(
            occupancy[i], temp_factor[i], element[i], charge[i]
        )
    return start, end

@cython.boundscheck(False)
@cython.wraparound(False)
def coordinate_record_lines(np.ndarray start, float[:, :] coords,
                            np.ndarray end):
    cdef size_t natoms = coords.shape[0]
    cdef size_t i
    cdef np.ndarray output = np.empty([natoms], dtype="U82")
    # assert inputs
    start = start.astype(S31)
    end = end.astype(S27)
    cdef float x, y, z
    for i in range(natoms):
        x = coords[i][0]
        y = coords[i][1]
        z = coords[i][2]
        output[i] = _sprintf_coord_line(start[i], x, y, z, end[i])
    return output.tolist()
