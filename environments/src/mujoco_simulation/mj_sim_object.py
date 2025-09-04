import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors as Nearest
import mujoco

from environments.src.mujoco_simulation.mj_client import MjClient
from environments.src.mj_search_space_bb_processor import get_search_space_bb_object

import configs.eval_config as eval_cfg

class MjSimObject:
    def __init__(self, mj_client: MjClient, name: str):

        self.object_name = name
        self.obj_id = name  # id associated with the object (for now I just use the same as the name) maybe later I dont need this
        self._path2obj_point_cloud = None
        self._obj_mesh_vertice_points = None
        self._list_of_points_for_each_triangle_obj_mesh = None
        self._search_space_bb_object = None
        self._obj_ss_bb_diagonal_dist = None
        self._uniform_obj_contact_points = None
        self._k_tree_uniform_contact_points = None

        self._init_attributes_mujoco(
            mj_client=mj_client,
            name=name
        )

    @property
    def path2obj_point_cloud(self):
        return self._path2obj_point_cloud

    @property
    def obj_mesh_vertice_points(self):
        return self._obj_mesh_vertice_points

    @property
    def object_normals_to_triangles(self):
        return self._object_normals_to_triangles

    @property
    def list_of_points_for_each_triangle_obj_mesh(self):
        return self._list_of_points_for_each_triangle_obj_mesh

    @property
    def search_space_bb_object(self):
        return self._search_space_bb_object

    @property
    def uniform_obj_contact_points(self):
        return self._uniform_obj_contact_points

    @property
    def k_tree_uniform_contact_points(self):
        return self._k_tree_uniform_contact_points

    # ========== NEW MUJOCO-BASED FUNCTIONS ==========
    
    def get_object_point_cloud(self, mj_client, body_name):
        """Get object geometry vertices"""
        return self._get_point_cloud(mj_client, body_name)

    def _get_point_cloud(self, mj_client, body_name):
        """
        Extract point cloud from MuJoCo geometry for a specific body.
        
        Returns:
            tuple: (precise_vertices_point, mesh, triangles) - same format as import_point_cloud_from_obj
        """
        model = mj_client.model
        
        # Get body ID
        try:
            body_id = model.body(body_name).id
        except:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
        
        # Find geom belonging to this body
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                geom_type = model.geom_type[geom_id]
                geom_size = model.geom_size[geom_id]
                
                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    # For mesh files, placeholder for now
                    raise NotImplementedError("Mesh file processing not yet implemented")
                else:
                    # For primitives, use Open3D
                    return self.create_primitive_mesh(geom_type, geom_size)
        
        # No geometry found for this body
        raise ValueError(f"No geometry found for body '{body_name}'")
    
    def create_primitive_mesh(self, geom_type, geom_size):
        """
        Create Open3D mesh for MuJoCo primitive shapes.
        
        Args:
            geom_type: MuJoCo geometry type 
            geom_size: MuJoCo geometry size parameters
            
        Returns:
            tuple: (precise_vertices_point, mesh, triangles)
        """
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            # Box: geom_size = [half_x, half_y, half_z]
            hx, hy, hz = geom_size[:3]
            mesh = o3d.geometry.TriangleMesh.create_box(width=2*hx, height=2*hy, depth=2*hz)
            
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            # Sphere: geom_size[0] = radius
            r = geom_size[0]
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20)
            
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # Cylinder: geom_size[0] = radius, geom_size[1] = half_height
            r, half_h = geom_size[0], geom_size[1]
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=2*half_h, resolution=30, split=4)
            
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # Capsule: geom_size[0] = radius, geom_size[1] = half_height
            r, half_h = geom_size[0], geom_size[1]
            mesh = o3d.geometry.TriangleMesh.create_capsule(radius=r, height=2*half_h, resolution=30, split=4)
            
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")
        
        mesh.compute_vertex_normals()
        
        # Extract vertices and triangles
        precise_vertices_point = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        return precise_vertices_point, mesh, triangles

    def extract_inertial_pose_from_mujoco(self, mj_client, body_name):
        """
        Extract inertial frame offset from MuJoCo body.
        In MuJoCo, this is the body's pose relative to its parent.
        
        Returns:
            np.array: [x, y, z] offset of inertial frame
        """
        model = mj_client.model
        
        try:
            body_id = model.body(body_name).id
        except:
            # If body not found, return zero offset
            return np.array([0.0, 0.0, 0.0])
        
        # Get body position relative to parent (inertial frame offset)
        body_pos = model.body_pos[body_id]
        return body_pos.copy()
    
    def shift_precise_vertices_point_based_on_inertia_mujoco(self, mj_client, precise_vertices_point, body_name):
        """
        Shift vertices from mesh frame to inertial frame using MuJoCo data.
        """
        inertial_pose = self.extract_inertial_pose_from_mujoco(mj_client, body_name)
        
        # Same logic as convert_mesh_pose_to_inertial_frame
        shifted_vertices_point = precise_vertices_point - inertial_pose
        return shifted_vertices_point

    def _init_attributes_mujoco(self, mj_client: MjClient, name):
        precise_vertices_point, mesh, triangles = self.get_object_point_cloud(mj_client, name)
        # For primitive objects, use the same geometry for visual as collision
        visual_mesh = mesh

        if not eval_cfg.LOAD_OBJECT_WITH_FIXED_BASE:
            precise_vertices_point = self.shift_precise_vertices_point_based_on_inertia_mujoco(
                mj_client, precise_vertices_point, name
            )

        self._search_space_bb_object = get_search_space_bb_object(mj_client.model, mj_client.data, object_name=name)
        
        self._obj_ss_bb_diagonal_dist = np.linalg.norm(
            [self._search_space_bb_object.aabb_max, self._search_space_bb_object.aabb_min]
        )

        self._obj_mesh_vertice_points = precise_vertices_point
        self._list_of_points_for_each_triangle_obj_mesh = self.explicit_triangles(
            triangles=triangles,
            precise_vertices_point=precise_vertices_point,
        )
        self._object_normals_to_triangles = self.find_normal_to_triangles_in_the_object(mesh)

        # Set up contact point sampling
        n_point_mesh_sample = self._get_n_point_mesh_sample()
        print(f'n_point_mesh_sample={n_point_mesh_sample}')
        sampled_point_cloud = np.asarray(visual_mesh.sample_points_uniformly(n_point_mesh_sample).points)
        shifted_sampled_point_cloud = self.shift_precise_vertices_point_based_on_inertia_mujoco(
                mj_client, sampled_point_cloud, name
            )
        self._uniform_obj_contact_points = shifted_sampled_point_cloud
        self._k_tree_uniform_contact_points = Nearest(n_neighbors=1, metric='minkowski').fit(shifted_sampled_point_cloud)

        print(f"Initialized MuJoCo object '{name}' with {len(precise_vertices_point)} vertices and {len(triangles)} triangles")

    def import_point_cloud_from_obj(self, path2obj_point_cloud):
        mesh = o3d.io.read_triangle_mesh(path2obj_point_cloud)
        array_vertices = np.asarray(mesh.vertices)
        array_triangle = np.asarray(mesh.triangles)
        return array_vertices, mesh, array_triangle

    def explicit_triangles(self, triangles, precise_vertices_point):
        list_of_points_in_a_triangle = []
        for t in range(len(triangles)):
            triangle_indexes = triangles[t]
            A_idx, B_idx, C_idx = triangle_indexes
            A_xyz = precise_vertices_point[A_idx].tolist()
            B_xyz = precise_vertices_point[B_idx].tolist()
            C_xyz = precise_vertices_point[C_idx].tolist()
            list_of_points_in_a_triangle.append([A_xyz, B_xyz, C_xyz])

        return list_of_points_in_a_triangle

    def find_normal_to_triangles_in_the_object(self, mesh):
        mesh.compute_vertex_normals()
        array_normals_to_triangles = np.asarray(mesh.triangle_normals)
        return array_normals_to_triangles

    def _get_n_point_mesh_sample(self):
        # defined ratio : 0.2m => 2000 points
        n_point_mesh_sample = int(self._obj_ss_bb_diagonal_dist * 2000 / 0.2)
        return n_point_mesh_sample