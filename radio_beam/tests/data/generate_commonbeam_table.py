
'''

UNUSED IN CURRENT TESTING SUITE!

Run in a CASA environment to generate the smallest common beam for a range of
parameters.
'''

RUN = False

if RUN:

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

    # im1 = ia.newimagefromarray(pixels=ia.makearray(0, [4, 4, 1, 4]))
    # for i, pa in enumerate([0, 20, 40, 60]):
    #     im1.setrestoringbeam(channel=i, major='4arcsec', minor='3arcsec',
    #                          pa="{}deg".format(pa))

    # im2 = ia.newimagefromarray(pixels=ia.makearray(0, [4, 4, 1, 4]))
    # for i, pa in enumerate([0, 60, 20, 40]):
    #     im2.setrestoringbeam(channel=i, major='4arcsec', minor='3arcsec',
    #                          pa="{}deg".format(pa))

    # im3 = ia.newimagefromarray(pixels=ia.makearray(0, [4, 4, 1, 4]))
    # for i, pa in enumerate([60, 40, 20, 0]):
    #     im3.setrestoringbeam(channel=i, major='4arcsec', minor='3arcsec',
    #                          pa="{}deg".format(pa))

    # im4 = ia.newimagefromarray(pixels=ia.makearray(0, [4, 4, 1, 4]))
    # for i, pa in enumerate([60, 0, 20, 40]):
    #     im4.setrestoringbeam(channel=i, major='4arcsec', minor='3arcsec',
    #                          pa="{}deg".format(pa))

    # print(im1.commonbeam())
    # print(im2.commonbeam())
    # print(im3.commonbeam())
    # print(im4.commonbeam())
