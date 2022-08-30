import healpy as hp

def load_map(map_fname, resolution=512):
    """
    Method loads in the Map in fits format, subtracts dipole 
    and downgrades it to a common resolution. 

    Returns 
    ------- 
    mp_i downgraded map
    """
    map_i    = hp.read_map(map_fname, field=(0,1,2))
    map_i[0] = hp.remove_dipole(map_i[0])
    map_i    = hp.ud_grade(map_i, resolution)

    return map_i
