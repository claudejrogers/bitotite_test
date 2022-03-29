# Biotite test
Does "cythonizing" inner loop string writing significantly improve performance
for `PDBFile.set_structure`?

On my system:
```
$ python test.py
numpy: 167.46 ms
cython: 40.35 ms
```