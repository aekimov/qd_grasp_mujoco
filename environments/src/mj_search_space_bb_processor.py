import numpy as np
from collections import namedtuple

SearchSpaceBB = namedtuple("SearchSpaceBB", "aabb_min aabb_max")

def get_search_space_bb(model, data, robot_name, object_name):
    """Main entry point - gets search space bounding box for object+robot"""
    object_plus_robot_bb = True
    if object_plus_robot_bb:
        return get_search_space_bb_object_plus_robot(model, data, robot_name, object_name)
    else:
        return get_search_space_bb_object(model, data, object_name)

def get_search_space_bb_side(ss_bb):
    """Gets the maximum side length of the search space bounding box"""
    all_ssbb_sides = np.array(ss_bb.aabb_max) - np.array(ss_bb.aabb_min)
    ssbb_side = all_ssbb_sides.max()
    return ssbb_side

def get_search_space_bb_object_plus_robot(model, data, robot_name, object_name):
    """Creates cubic search space centered on object, sized to contain both robot and object"""
    robot_aabb_min, robot_aabb_max = big_boundary_box(model, data, robot_name)
    obj_aabb_min, obj_aabb_max = big_boundary_box(model, data, object_name)
    
    # Sum the bounding boxes (like adding their extents)
    aabb_min_tuple = np.array([robot_aabb_min, obj_aabb_min]).sum(axis=0)
    aabb_max_tuple = np.array([robot_aabb_max, obj_aabb_max]).sum(axis=0)
    
    # Find maximum dimension to make it cubic
    l_max = (np.array(aabb_max_tuple) - np.array(aabb_min_tuple)).max()
    
    # Center the cubic box on the object
    bb_centroid = (np.array(obj_aabb_min) + np.array(obj_aabb_max)) / 2
    
    # Create cubic bounds
    min_x, max_x = bb_centroid[0] - l_max / 2, bb_centroid[0] + l_max / 2
    min_y, max_y = bb_centroid[1] - l_max / 2, bb_centroid[1] + l_max / 2
    min_z, max_z = bb_centroid[2] - l_max / 2, bb_centroid[2] + l_max / 2
    aabb_min = [min_x, min_y, min_z]
    aabb_max = [max_x, max_y, max_z]
    
    ss_bb = SearchSpaceBB(aabb_min=aabb_min, aabb_max=aabb_max)
    return ss_bb

def get_search_space_bb_object(model, data, object_name):
    """Gets search space for just the object"""
    aabb_min, aabb_max = get_body_aabb(model, data, object_name)
    ss_bb = SearchSpaceBB(aabb_min=list(aabb_min), aabb_max=list(aabb_max))
    return ss_bb

def big_boundary_box(model, data, body_name):
    """Creates a cubic bounding box around a body (matches PyBullet version's behavior)"""
    aabb_min_tuple, aabb_max_tuple = get_body_aabb(model, data, body_name)
    
    # Make it cubic: find max dimension
    l_max = (np.array(aabb_max_tuple) - np.array(aabb_min_tuple)).max()
    bb_centroid = (np.array(aabb_max_tuple) + np.array(aabb_min_tuple)) / 2
    
    # Create cubic bounds centered on the body
    min_x, max_x = bb_centroid[0] - l_max / 2, bb_centroid[0] + l_max / 2
    min_y, max_y = bb_centroid[1] - l_max / 2, bb_centroid[1] + l_max / 2
    min_z, max_z = bb_centroid[2] - l_max / 2, bb_centroid[2] + l_max / 2
    aabb_min = [min_x, min_y, min_z]
    aabb_max = [max_x, max_y, max_z]
    
    return aabb_min, aabb_max

def is_descendant_body(model, parent_id, body_id):
    """Check if body_id is a descendant of parent_id (or is parent_id itself)"""
    if body_id == parent_id:
        return True
    
    # Traverse up the parent chain from body_id
    current = body_id
    while current != 0:  # 0 is world body
        current = model.body_parentid[current]
        if current == parent_id:
            return True
        if current == 0:  # reached world body
            break
    
    return False

def get_body_aabb(model, data, body_name):
    """Gets the world-space AABB for all geoms belonging to a body and its descendants"""
    # Get body ID
    body_id = model.body(body_name).id
    
    # Collect all geoms that belong to this body or its descendants
    geom_ids = []
    for geom_id in range(model.ngeom):
        geom_body_id = model.geom_bodyid[geom_id]
        # Check if this geom's body is our target body or a descendant
        if is_descendant_body(model, body_id, geom_body_id):
            geom_ids.append(geom_id)
    
    if not geom_ids:
        # If no geoms found, return a small box at body position
        body_pos = data.xpos[body_id]
        return body_pos - 0.01, body_pos + 0.01
    
    # Get world-space bounds for all collected geoms
    all_min = []
    all_max = []
    
    for geom_id in geom_ids:
        # Get geom AABB (center, half-sizes) in geom's local frame
        aabb = model.geom_aabb[geom_id]
        center_local = aabb[:3]
        half_sizes = aabb[3:]
        
        # Transform center to world frame
        # geom_xpos is the position of the geom frame origin in world
        # We need to add the rotated local center offset
        R = data.geom_xmat[geom_id].reshape(3, 3)
        center_world = data.geom_xpos[geom_id] + R @ center_local
        
        # For an axis-aligned bounding box in world space, we need to
        # consider how the rotated box extends in each world axis
        # This is done by summing the absolute values of rotated half-extents
        world_half_sizes = np.abs(R) @ half_sizes
        
        # World-space AABB
        geom_min = center_world - world_half_sizes
        geom_max = center_world + world_half_sizes
        
        all_min.append(geom_min)
        all_max.append(geom_max)
    
    # Find overall bounds
    aabb_min = np.array(all_min).min(axis=0)
    aabb_max = np.array(all_max).max(axis=0)
    
    return aabb_min, aabb_max