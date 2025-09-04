# import xml.etree.ElementTree as ET
# from pathlib import Path
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors as Nearest
import mujoco

from environments.src.mujoco_simulation.mj_client import MjClient
from environments.src.mj_search_space_bb_processor import get_search_space_bb_object
# from algorithms.evaluation.grasp_strategies_routines import convert_mesh_pose_to_inertial_frame

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
    
    def get_collision_point_cloud(self, mj_client, body_name):
        """Get collision geometry vertices (group 3)"""
        return self._get_point_cloud(mj_client, body_name, group=3)

    def get_visual_point_cloud(self, mj_client, body_name):
        """Get visual geometry vertices (group 2)"""
        return self._get_point_cloud(mj_client, body_name, group=2)
    
    def get_object_point_cloud(self, mj_client, body_name):
        """Get object geometry vertices (group 5)"""
        return self._get_point_cloud(mj_client, body_name, group=5)

    def _get_point_cloud(self, mj_client, body_name, group):
        """
        Extract point cloud from MuJoCo geometry for a specific body and group.
        
        Returns:
            tuple: (precise_vertices_point, mesh, triangles) - same format as import_point_cloud_from_obj
        """
        model = mj_client.model
        
        # Get body ID
        try:
            body_id = model.body(body_name).id
        except:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
        
        # Find geom belonging to this body and group
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id and model.geom_group[geom_id] == group:
                geom_type = model.geom_type[geom_id]
                geom_size = model.geom_size[geom_id]
                
                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    # For mesh files, placeholder for now
                    raise NotImplementedError("Mesh file processing not yet implemented")
                else:
                    # For primitives, use Open3D
                    return self.create_primitive_mesh(geom_type, geom_size)
        
        # No geometry found for this group
        raise ValueError(f"No geometry found for body '{body_name}' in group {group}")
    
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

    # ========== OLD URDF-BASED FUNCTIONS (for reference) ==========

    # def _init_attributes(self, mj_client: MjClient, name):
    #     path2obj_point_cloud = self.extract_path2obj_contact_point_cloud_from_urdf(object_name=name)
    #     precise_vertices_point, mesh, triangles = self.import_point_cloud_from_obj(path2obj_point_cloud)

    #     path2obj_point_cloud_visual = self.extract_path2obj_visual_point_cloud_from_urdf(object_name=name)
    #     _, visual_mesh, _ = self.import_point_cloud_from_obj(path2obj_point_cloud_visual)

    #     if not eval_cfg.LOAD_OBJECT_WITH_FIXED_BASE:
    #         precise_vertices_point = self.shift_precise_vertices_point_based_on_inertia(precise_vertices_point)

    #     self._search_space_bb_object = get_search_space_bb_object(obj_id=self.obj_id)
    #     self._obj_ss_bb_diagonal_dist = np.linalg.norm(
    #         [self._search_space_bb_object.aabb_max, self._search_space_bb_object.aabb_min]
    #     )

    #     self._obj_mesh_vertice_points = precise_vertices_point
    #     self._list_of_points_for_each_triangle_obj_mesh = self.explicit_triangles(
    #         triangles=triangles,
    #         precise_vertices_point=precise_vertices_point,
    #     )
    #     self._object_normals_to_triangles = self.find_normal_to_triangles_in_the_object(mesh)

    #     n_point_mesh_sample = self._get_n_point_mesh_sample()
    #     print(f'n_point_mesh_sample={n_point_mesh_sample}')
    #     sampled_point_cloud = np.asarray(visual_mesh.sample_points_uniformly(n_point_mesh_sample).points)
    #     shifted_sampled_point_cloud = self.shift_precise_vertices_point_based_on_inertia(sampled_point_cloud)
    #     self._uniform_obj_contact_points = shifted_sampled_point_cloud

    #     self._k_tree_uniform_contact_points = Nearest(n_neighbors=1, metric='minkowski')
    #     self._k_tree_uniform_contact_points.fit(shifted_sampled_point_cloud)

    # def get_object_root_path(self, object_name):
    #     return Path(__file__).parent.parent.parent.parent/"3d_models/objects"/object_name

    # def get_path_to_urdf(self, object_name):
    #     return self.get_object_root_path(object_name)/f"{object_name}.urdf"

    # def extract_path2obj_contact_point_cloud_from_urdf(self, object_name):
    #     path_to_urdf_obj = self.get_path_to_urdf(object_name)
    #     root_path_to_obj = str(self.get_object_root_path(object_name))

    #     path2obj_point_cloud = None
    #     tree = ET.parse(path_to_urdf_obj)
    #     obj_root = tree.getroot()[0]

    #     for element in obj_root:
    #         if 'collision' not in element.tag:
    #             continue

    #         for collision_element in element:

    #             if 'geometry' not in collision_element.tag:
    #                 continue

    #             mesh_element = collision_element[0]
    #             assert mesh_element.tag == 'mesh'
    #             path2obj_point_cloud = f"{root_path_to_obj}/{mesh_element.attrib['filename']}"

    #     assert path2obj_point_cloud is not None, \
    #         f'Invalid urdf structure : cannot extract contact path to point cloud from {path_to_urdf_obj}.'

    #     return path2obj_point_cloud

    # def extract_path2obj_visual_point_cloud_from_urdf(self, object_name):
    #     path_to_urdf_obj = self.get_path_to_urdf(object_name)
    #     root_path_to_obj = str(self.get_object_root_path(object_name))

    #     path2obj_point_cloud = None
    #     tree = ET.parse(path_to_urdf_obj)
    #     obj_root = tree.getroot()[0]

    #     for element in obj_root:
    #         if 'visual' not in element.tag:
    #             continue

    #         for collision_element in element:

    #             if 'geometry' not in collision_element.tag:
    #                 continue

    #             mesh_element = collision_element[0]
    #             assert mesh_element.tag == 'mesh'
    #             path2obj_point_cloud = f"{root_path_to_obj}/{mesh_element.attrib['filename']}"

    #     assert path2obj_point_cloud is not None, \
    #         f'Invalid urdf structure : cannot extract contact path to point cloud from {path_to_urdf_obj}.'

    #     return path2obj_point_cloud

    # def extract_inertial_pose_from_urdf(self, object_name):

    #     path_to_urdf_obj = self.get_path_to_urdf(object_name)

    #     inertial_pose = None
    #     tree = ET.parse(path_to_urdf_obj)
    #     obj_root = tree.getroot()[0]

    #     for element in obj_root:
    #         if 'inertial' not in element.tag:
    #             continue

    #         for inertial_element in element:
    #             if 'origin' not in inertial_element.tag:
    #                 continue

    #             inertial_pose_str = inertial_element.attrib['xyz'].split(' ')
    #             inertial_pose = np.array([float(pose_str) for pose_str in inertial_pose_str])

    #     return inertial_pose

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

    # def shift_precise_vertices_point_based_on_inertia(self, precise_vertices_point):

    #     inertial_pose = self.extract_inertial_pose_from_urdf(self.object_name)
    #     shifted_vertices_point = convert_mesh_pose_to_inertial_frame(
    #         obj_inertial_pose=inertial_pose, meshpose2cvt=precise_vertices_point
    #     )
    #     return shifted_vertices_point

    def _get_n_point_mesh_sample(self):
        # defined ratio : 0.2m => 2000 points
        n_point_mesh_sample = int(self._obj_ss_bb_diagonal_dist * 2000 / 0.2)
        return n_point_mesh_sample
    
    
    
    # def _load_object_bullet(self, bullet_client):
    #     self.load_object(
    #         bullet_client=bullet_client,
    #         object_name=self.object_name
    #     )

    # def _init_object_name(self, object_name):
    #     return object_name.strip() if object_name is not None else None

    # def load_object(self, bullet_client, object_name=None):
    #     assert isinstance(object_name, str)

    #     urdf = self.get_path_to_urdf(object_name)
    #     if not urdf.exists():
    #         raise ValueError(str(urdf) + " doesn't exist")

    #     try:
    #         obj_to_grab_id = bullet_client.loadURDF(
    #             fileName=str(urdf),
    #             basePosition=DEFAULT_OBJECT_POSE,
    #             baseOrientation=DEFAULT_OBJECT_ORIENT,
    #             useFixedBase=eval_cfg.LOAD_OBJECT_WITH_FIXED_BASE
    #         )
    #     except bullet_client.error as e:
    #         raise bullet_client.error(f"{e}: " + str(urdf))

    #     bullet_client.changeDynamics(
    #         bodyUniqueId=obj_to_grab_id,
    #         linkIndex=-1,
    #         spinningFriction=eval_cfg.SPINNING_FRICTION_OBJ_DEFAULT_VALUE,
    #         rollingFriction=eval_cfg.ROLLING_FRICTION_OBJ_DEFAULT_VALUE,
    #     )

    #     self.obj_id = obj_to_grab_id
    
    
    # def update_infos(self, bullet_client, info):
    #     is_obj_initialized = self.obj_id is not None
    #     if is_obj_initialized:
    #         obj_pose = bullet_client.getBasePositionAndOrientation(self.obj_id)
    #         obj_vel = bullet_client.getBaseVelocity(self.obj_id)
    #         info['object position'], info['object xyzw'] = obj_pose
    #         info['object linear velocity'], info['object angular velocity'] = obj_vel
    #     return info