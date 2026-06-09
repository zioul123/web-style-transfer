import drjit
import mitsuba as mi
import trimesh

mi.set_variant("llvm_ad_rgb")

import torch
import torch_geometric.data
import torch_geometric.transforms
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.nn import knn_interpolate

def define_camera(
    camera_distance: float,
    azimuth_deg: float,
    elevation_deg: float,
    camera_type: str = "perspective",
    img_width: int = 1024,
    img_height: int = 1024,
    sampler_type: str = "multijitter",  # default was "independent"
    sample_count: int = 16,
    fov: float = 40,
    aperture_radius: float | None = None,
    focus_distance: float | None = None,
    camera_pos_override: list[int] | None = None,
    camera_up_override: list[int] | None = None,
    target: list[int] = [0, 0, 0],
) -> dict:
    camera_pos = mi.ScalarTransform4f.rotate([0, 0, 1], elevation_deg).rotate(
        [0, 1, 0], azimuth_deg
    ) @ mi.ScalarPoint3f([0, 0, camera_distance])
    camera = {
        "type": camera_type,
        "fov": fov,
        "near_clip": 0.01,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.look_at(
            origin=(camera_pos if camera_pos_override is None else camera_pos_override), 
            target=target, 
            up=([0, 1, 0] if camera_up_override is None else camera_up_override)
        ),
        "film": {
            "type": "hdrfilm",
            "rfilter": {"type": "box"},
            "width": img_width,
            "height": img_height,
        },
        "sampler": {
            "type": sampler_type,
            "sample_count": sample_count,
        },
    }
    if camera_type == "thinlens":
        camera["aperture_radius"] = aperture_radius
        camera["focus_distance"] = focus_distance
    return camera

class PclColoursTexture(mi.Texture):
    """
    Python plugin for mitsuba 3. It allows to store a texture as a point cloud
    instead of as an image. Rays intersecting the surface search for the 3
    nearest neighbours on the point cloud and interpolate their colours to
    determine the colour at the ray intersection.

    This plugin clushes with the efficient mitsuba implementation. Therefore,
    before rendering the megakernel needs to be shut down: call
    mega_kernel(state=False) before rendering.

    The main disadvantage of disabling the megakernel is the GPU memory
    consumption which increases significantly (and remains high even after
    rendering). Flush the cache with flush_cache() after rendering. Given the
    high memory consumption, you may have to flush the torch cache even before
    rendering.
    """

    def __init__(self, props: mi.Properties) -> None:
        mi.Texture.__init__(self, props)
        self._grad_activator = mi.Vector3f(0)
        self.pcl_torch_pos = None
        self.pcl_mi_cols = None

    def traverse(self, callback):
        callback.put(
            "grad_activator", self._grad_activator, mi.ParamFlags.Differentiable
        )
        callback.put(
            "pcltex_pos", self.pcl_torch_pos, mi.ParamFlags.NonDifferentiable
        )
        callback.put(
            "pcltex_color", self.pcl_mi_cols, mi.ParamFlags.Differentiable
        )

    def eval(self, si, active=True, dirs=None, norms=None, albedo=None):
        surface_intersection_position = vec_to_tens_safe(si.p)
        mi_out = self._eval_in_torch(
            surface_intersection_position, self.pcl_mi_cols
        )
        return drjit.unravel(mi.Vector3f, drjit.ravel(mi_out))

    @drjit.wrap(source="drjit", target="torch")
    def _eval_in_torch(self, pts, pcl_cols):
        # Find k-NN of pcl_torch_pos to pts with k=3 and interpolate colour
        # from colour of 3-NN

        interpolated_cols_torch = knn_interpolate(
            pcl_cols.to(pts.device),
            self.pcl_torch_pos.to(pts.device),
            pts,
            k=3,
        )
        return interpolated_cols_torch

    def eval_1(self, si, active=True):
        return mi.Float(self.eval(si)[0])

    def eval_1_grad(self, *args, **kwargs):
        raise NotImplementedError()

    def eval_3(self, *args, **kwargs):
        raise NotImplementedError()

    def mean(self, *args, **kwargs):
        raise NotImplementedError()

    def to_string(self):
        return "PclColoursTexture"


mi.register_texture("pcl_colours_texture", lambda p: PclColoursTexture(p))


def vec_to_tens_safe(vec):
    # A utility function that converts a Vector3f to a TensorXf safely in
    # mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(
        drjit.ravel(vec), shape=(drjit.shape(vec)[1], drjit.shape(vec)[0])
    )

def mega_kernel(state: bool = False):
    drjit.set_flag(drjit.JitFlag.LoopRecord, state)
    drjit.set_flag(drjit.JitFlag.VCallRecord, state)
    drjit.set_flag(drjit.JitFlag.VCallOptimize, state)


def flush_cache():
    for _ in range(5):  # Not sure why but calling it once is not enough
        drjit.flush_malloc_cache()


def mesh_with_pcltex_to_mitsuba(
    data: torch_geometric.data.Data,
    normalise_scale: bool = False,
    twosided: bool = True,
) -> mi.Mesh:
    data = data.detach().cpu()

    if normalise_scale:
        data = torch_geometric.transforms.NormalizeScale()(data)

    verts = np.array(data.verts)
    faces = np.array(data.face)
    pcl_cols = data.x.clamp(0, 1)
    pcl_colours_texture = mi.load_dict({"type": "pcl_colours_texture"})
    pcl_colours_texture.pcl_torch_pos = data.pos

    pcl_colours_texture.pcl_mi_cols = mi.TensorXf(
        drjit.ravel(mi.TensorXf(pcl_cols.squeeze())),
        shape=pcl_cols.squeeze().shape,
    )

    bsdf_dict = {
        "type": "principled",
        "base_color": pcl_colours_texture,
    }
    if twosided:
        bsdf_dict = {"type": "twosided", "material": bsdf_dict}

    bsdf_prop = mi.Properties()
    bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

    mi_mesh = mi.Mesh(
        "mesh",
        vertex_count=verts.shape[0],
        face_count=faces.shape[0],
        props=bsdf_prop,
    )

    # "Traverse" the mesh to get its updateable parameters
    mesh_params = mi.traverse(mi_mesh)
    mesh_params["vertex_positions"] = verts.flatten()
    mesh_params["faces"] = faces.flatten()
    mesh_params.update()
    return mi_mesh


def load_mesh(
    file_path: str, show: bool = False, merge_tex: bool = True
) -> trimesh.Trimesh:
    scene = trimesh.load(file_path, process=False)

    if hasattr(scene, "graph"):
        geometries = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            # get a copy of the geometry
            current = scene.geometry[geometry_name].copy()
            if isinstance(current, trimesh.Trimesh):
                # move the geometry vertices into the requested frame
                try:
                    current.apply_transform(transform)
                except RuntimeWarning:
                    print(f"troubles with {file_path}")

                # If there are pre-existing uvs in regions with a uniform colour
                # and no texture the visual concatenation fails.
                # Delete those uvs!
                try:
                    if current.visual.material.baseColorTexture is None:
                        current.visual.uv = None
                except AttributeError:
                    if current.visual.material.image is None:
                        current.visual.uv = None

                # save to our list of meshes
                geometries.append(current)

        if len(geometries) > 1:
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = geometries[0]
    else:
        mesh = scene

    trimesh.grouping.merge_vertices(mesh, merge_tex=merge_tex, merge_norm=True)

    if show:
        mesh.show()
    return mesh

if __name__ == "__main__":
    import os
    import torch

    import mitsuba as mi

    mi.set_variant("llvm_ad_rgb")

    import rendering

    root = "."
    data_path = os.path.join(root, "flattoflatfemme.pt")
    data = torch.load(data_path, weights_only=False)

    # position = tuple(np.array([0, 0.8, 1])*0.4)    # Front
    # position = tuple(np.array([0.4, 0.8, -1])*0.4) # Back
    # position = tuple(np.array([0.4, 0.6, 0])*0.65) # Right
    # focal_point = [-0.025, 0.1, 0]
    # position = tuple(np.array([-0.4, 0.2, 0.2])*0.9) # Left
    # focal_point = [-0.025, 0.1, -0.01]

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator":  {"type": "path", "hide_emitters": False},
            # Good for plane
            "camera": define_camera(4, 30, 80,  # dummy
                                    img_height=512, 
                                    img_width=512,
                                    camera_pos_override=[0, 3, 0],
                                    camera_up_override=[0, 0, -1]),
            # "camera": define_camera(0.5, 30, 80,
            #                         img_height=512,
            #                         img_width=512,
            #                         target=[-0.025, 0.1, 0],
            #                         camera_pos_override=[0, 0.24, 0.3],
            #                         camera_up_override=[0, 0, -1]),
            # "camera": define_camera(10, 90, 0),
            "emitter": { "type": "constant" },
            "mesh": mesh_with_pcltex_to_mitsuba(data),
        }
    )
    mega_kernel(False)                         
    image = mi.render(scene)
    plt.axis("off")
    plt.imshow(image)
    plt.savefig("abc.png")
    flush_cache()
