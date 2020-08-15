def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': ['data/*.fits.gz',
                                            'data/*.tar.gz',
                                            'data/*hdr',
                                            'data/*.csv']}
