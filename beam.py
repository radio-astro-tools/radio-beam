from astropy import units as u

class Beam(u.Unit):
    """ 
    """
    pass

    @property
    def minor():
        ...
        
    @property
    def major():
        ...

    @property
    def area():
        return self.minor * self.major
        
