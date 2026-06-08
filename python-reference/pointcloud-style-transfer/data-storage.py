from torch_geometric.data import Data

def create_data(
    pc_positions,  # [N, 3]
    pc_colors,     # [N, 3]
    mesh_vertices, # [V, 3]
    mesh_faces     # [F, 3]
):
    return Data(x=pc_colors,
                pos=pc_positions,
                verts=mesh_vertices,
                face=mesh_faces)
  