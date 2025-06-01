import numpy as np
import uaibot as ub


def check_free(_p, _pc, _radius, _bounds):
    
    if _p[0,0]<_bounds[0] or _p[0,0]>_bounds[1] or \
       _p[1,0]<_bounds[2] or _p[1,0]>_bounds[3] or \
       _p[2,0]<_bounds[4] or _p[2,0]>_bounds[5]:
           return False
    else:

        return _pc.projection(_p)[1]>1.2*_radius
    
def sample_random_direction(_p, _p_goal, _step_size):
    
    if np.random.uniform()<0.5:
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)
        return _step_size* np.matrix(vec).T
    else:
        return _p_goal-_p

def is_line_free(_p1, _p2, _pc, _radius, _bounds):
    dist = np.linalg.norm(_p1-_p2)
    no_steps = int(dist/0.05)+1
    for i in range(no_steps):
        alpha = i/(no_steps-1+0.01)
        point = (1 - alpha) * _p1 + alpha * _p2
        if not check_free(point, _pc, _radius, _bounds):
            return False
    return True

def simplify_path(_path, _pc, _radius, _bounds):
    if len(_path) <= 2:
        return _path

    simplified = [_path[0]]
    i = 0
    while i < len(_path) - 1:
        j = len(_path) - 1
        while j > i + 1:
            if is_line_free(_path[i], _path[j], _pc, _radius, _bounds):
                break
            j -= 1
        simplified.append(_path[j])
        i = j

    return simplified

def random_path_planner(_p_start, _p_goal, _pc, _radius, _bounds,  _max_steps=12000, _step_size=0.5):
    path = [_p_start]
    current = _p_start.copy()

    found = False
    
    
    for i in range(_max_steps):
        direction = sample_random_direction(current, _p_goal, _step_size)
        new_point = current + direction 

        # Check line from current to new_point is free
        if not is_line_free(current, new_point, _pc, _radius, _bounds):
            continue

        path.append(new_point)
        
        current = new_point

        # Try to connect to goal if close enough
        if np.linalg.norm(_p_goal - current) < _step_size:
            if is_line_free(current, _p_goal, _pc, _radius, _bounds):
                path.append(_p_goal)
                found = True
                break

    if found:
        simpl_path = simplify_path(path, _pc, _radius, _bounds)
        print("Path found with "+str(len(simpl_path)-1))
        return simpl_path[1:]
    else:
        print("Path not found!")
        return [_p_goal]



