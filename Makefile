all:
	python setup.py build_ext --inplace

html:
	cython -a --3str src/pdbhelper.pyx

clean:
	rm -r build
	rm src/*.c
	rm src/*.html
	rm *.so 