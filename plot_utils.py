import numpy as np

def Ammonia_projection(xyzs):
    """ Returns the two degrees of freedom for a given set of coordinates.
    XYZS are assumed to be of shape (4,3) and in order: N, H, H, H.
    First coordinate q1 is the distance of the N atom to the plane spanned by
    the three H atoms. Second coordiate q2 is the mean of the distances between
    the H atoms and the N atom.
    """

    # normal vector:
    n = np.cross(xyzs[2,:]-xyzs[1,:], xyzs[3,:]-xyzs[1,:])
    # intercept in point-normal form (ax + by + bc + d = 0)
    d = -np.dot(n, xyzs[1,:])
    # q1
    q1 = (np.dot(n, xyzs[0,:]) + d)/np.linalg.norm(n)
    # q2
    q2 = np.mean(np.linalg.norm(xyzs[1:,:] - xyzs[0,:], axis = 1))
    return q1, q2

def Ammonia_geometry_from_qs(q1, q2):
    """ Converts the generalized coordinates q1 and q2 into a (4,3) dimensional
    array of atomic coordinates. The H atoms are positioned in the xy plane,
    the N atom on the z axis.
    """
    xyzs = np.zeros((4,3))
    # N atom in z direction
    xyzs[0,2] = q1
    # First H atom in x direction
    xyzs[1,0] = np.sqrt(q2**2 - q1**2)
    # Second H atom in a direction rotated 120 degrees from the x direction.
    # The x coordinate is x = cos(120) = -1/2. The y coordinate is y = sin(120)
    # = sqrt(3)/2. Both components are rescaled to ensure the given NH distance
    # q1.
    xyzs[2,0] = -np.sqrt(q2**2 - q1**2)/2
    xyzs[2,1] = np.sqrt(3*(q2**2 - q1**2))/2
    # Third H atom in 240 degrees from the x direction. Calculated same as the
    # second H atom but with negative y component.
    xyzs[3,0] = -np.sqrt(q2**2 - q1**2)/2
    xyzs[3,1] = -np.sqrt(3*(q2**2 - q1**2))/2
    return xyzs

def Ethane_projection(xyzs):
    """ Converts the xyz coordinates of the Ethane molecule into two generalized
    coordinates q1 and q2. The first coordinate q1 is the dihedral angle (in
    degrees) between two H atoms on the different methyl fragment (between the
    first atom and the sixth atom). Second coordinate q2 for now is the C=C
    distance, but might be replaced with a more sofisticated choice at some
    later point. Expects an (8,3) array in the order H, H , H , C, C, H, H, H.
    """
    # First construct the normal vectors of the H1,C1,C2 and H4,C2,C1 planes.
    n1 = np.cross(xyzs[0,:]-xyzs[3,:], xyzs[4,:]-xyzs[3,:])
    n2 = np.cross(xyzs[3,:]-xyzs[4,:], xyzs[5,:]-xyzs[4,:])
    # q2 is the angle between the two planes.
    q1 = (np.arccos(np.dot(n1,n2)/(np.linalg.norm(n1)*np.linalg.norm(n2)))
        *180.)/np.pi
    # q2 == CC distance
    q2 = np.linalg.norm(xyzs[4,:]-xyzs[3,:])

    return q1, q2

def Ethane_geometry_from_qs(q1, q2):

    CH_distance = 1.086
    # CH_angle
    CH_angle = 110.7 *np.pi/180.
    xyzs = np.zeros((8,3))

    # Place the C atoms along the z axis with distance q2
    xyzs[3,2] = q2/2
    xyzs[4,2] = -q2/2
    # First methly group is placed in fixed rotation
    xyzs[0,0] = CH_distance*np.sin(CH_angle)
    xyzs[0,2] = q2/2 - CH_distance*np.cos(CH_angle)

    xyzs[1,0] = CH_distance*np.sin(CH_angle)*np.cos((120.)*np.pi/180.)
    xyzs[1,1] = -CH_distance*np.sin(CH_angle)*np.sin((120.)*np.pi/180.)
    xyzs[1,2] = q2/2 - CH_distance*np.cos(CH_angle)

    xyzs[2,0] = CH_distance*np.sin(CH_angle)*np.cos((240.)*np.pi/180.)
    xyzs[2,1] = -CH_distance*np.sin(CH_angle)*np.sin((240.)*np.pi/180.)
    xyzs[2,2] = q2/2 - CH_distance*np.cos(CH_angle)

    xyzs[5,0] = CH_distance*np.sin(CH_angle)*np.cos(q1*np.pi/180.)
    xyzs[5,1] = -CH_distance*np.sin(CH_angle)*np.sin(q1*np.pi/180.)
    xyzs[5,2] = -q2/2 + CH_distance*np.cos(CH_angle)

    xyzs[6,0] = CH_distance*np.sin(CH_angle)*np.cos((q1+120.)*np.pi/180.)
    xyzs[6,1] = -CH_distance*np.sin(CH_angle)*np.sin((q1+120.)*np.pi/180.)
    xyzs[6,2] = -q2/2 + CH_distance*np.cos(CH_angle)

    xyzs[7,0] = CH_distance*np.sin(CH_angle)*np.cos((q1+240.)*np.pi/180.)
    xyzs[7,1] = -CH_distance*np.sin(CH_angle)*np.sin((q1+240.)*np.pi/180.)
    xyzs[7,2] = -q2/2 + CH_distance*np.cos(CH_angle)

    return xyzs
