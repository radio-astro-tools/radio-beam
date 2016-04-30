all: advs.fits dvsa.fits vsad.fits sadv.fits sdav.fits adv.fits vad.fits

%.fits:
	python -c "import runpy;\
	           from spectral_cube.tests.data import make_test_cubes;\
			   runpy.run_path(make_test_cubes.__file__)"

