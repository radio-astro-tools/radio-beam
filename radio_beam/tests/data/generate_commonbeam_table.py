
'''
Run in a CASA environment to generate the smallest common beam for a range of
parameters.
'''

try:
    import casac
except ImportError:
    raise ImportError("This script must be run in CASA")

# Tested for CASA 4.7.2
import casadef
if casadef.casa_version != "4.7.2":
    raise Exception("This script is only tested in CASA 4.7.2. "
                    "Found version {}".format(casadef.casa_version))

import numpy as np

# Generate fake cube w/ 2 spectral channels
im1 = ia.newimagefromarray(pixels=ia.makearray(0, [4, 4, 1, 2]))

# Give the first channel a circular beam
im1.setrestoringbeam(major='3arcsec', minor='3arcsec', pa='0deg', channel=0)


# Give the other channel an elongated beam, and rotate it
majors = []
minors = []
pas = []

# Include the PA of the one beam so tests can be run later.
orig_pas = np.arange(179)

for pa in orig_pas:
    im1.setrestoringbeam(major='4arcsec', minor='2.5arcsec',
                         pa='{}deg'.format(pa), channel=1)

    com_beam = im1.commonbeam()

    majors.append(com_beam['major']['value'])
    minors.append(com_beam['minor']['value'])
    pas.append(com_beam['pa']['value'])

common_beams = np.array([orig_pas, majors, minors, pas]).T
np.savetxt("commonbeam_CASA_comparison.csv", common_beams, delimiter=",")
