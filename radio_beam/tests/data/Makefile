all: advs.fits

%.fits:
	python -c "import runpy;\
	           from spectral_cube.tests.data import make_test_cubes;\
			   runpy.run_path(make_test_cubes.__file__,\
			                  run_name='__main__')"

