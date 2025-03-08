import numpy as np

def closest_point_on_simplex(simplex):
    simplex = np.array(simplex)  

    if len(simplex) == 1:  
        return simplex[0], simplex

    elif len(simplex) == 2:  
        A, B = simplex
        AB = B - A
        AO = -A  

        t = np.dot(AO, AB) / np.dot(AB, AB)

        if t < 0:
            return A, [A]  
        elif t > 1:
            return B, [B]  
        else:
            closest_point = A + t * AB  
            return closest_point, [A, B]

    elif len(simplex) == 3:  
        A, B, C = simplex
        AO = -A
        AB = B - A
        AC = C - A

        normal = np.cross(AB, AC)

        AB_perp = np.array([-AB[1], AB[0]])  
        AC_perp = np.array([-AC[1], AC[0]])  

        if np.dot(AB_perp, AO) > 0:  
            return closest_point_on_simplex([A, B])  
        elif np.dot(AC_perp, AO) > 0:  
            return closest_point_on_simplex([A, C])  
        else:
            return A, [A]  
