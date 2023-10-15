import numpy as np
import cv2
import gmsh
import meshio

class bitmapMeshHelper:
    def __init__(self, data: np.ndarray, scale: float=1., origin: np.ndarray=np.zeros((1,2))):
        self.data = data
        self.scale = scale
        self.origin = origin
    
    def getCoordinates(self, data):
        """Get coordinates of bitmap image pixels.
        
        Args:
            data (np.ndarray): numpy array of bitmap image data.

        Returns:
            np.ndarray: xy coordinates of size (npts, 2)
        """

        # grab coordinates of bitmap and scale.
        xy_coords = np.flip(np.column_stack(np.where(data > 0)), axis=1)
        xy_coords = xy_coords * self.scale

        # translate image to new origin.
        xy_coords = self.translate(xy_coords)
        
        return xy_coords
        
    def translate(self, xy_coords: np.ndarray) -> np.ndarray:
        """Translate image to be centered about a point.

        Args:
            xy_coords (np.ndarray): (x, y) coordinates to be translated

        Returns:
            np.ndarray: Translated coordinates centered about the midpoint of `self.data`.
        """
        center = self._computeCenter()
        xy_coords = np.subtract(xy_coords, center)
        xy_coords = np.add(xy_coords, self.origin)
        return xy_coords

    def rotate(self, angle: float):
        raise NotImplementedError

    def _computeCenter(self) -> np.ndarray:
        """Compute the center of the image from data dimensions.
        NOTE that this does not compute the centroid of the supplied data.

        Returns:
            np.ndarray: (x, y) coordinates of image center.
        """
        return np.flip(np.array(self.data.shape)/2).astype(int) * self.scale

    def _boundary_opencv(self):
        """Use OpenCV to generate ordered boundary.

        Returns:
            np.ndarray: Ordered points on Boundary.
        """
        contours = cv2.findContours(self.data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        cntr = contours[0]
        boundary = cntr.squeeze()

        # scale and translate.
        boundary = boundary * self.scale
        boundary = self.translate(xy_coords=boundary)

        return boundary
    
    def _boundary_numpy(self):
        """Use pure NumPy to generate ordered boundary using angle to center to order the points.

        Args:
            xy_coords (np.ndarray): xy coordinates of pixel data.

        Returns:
            np.ndarray: Ordered points on boundary.
        """
        grad = np.gradient(self.data)
        boundary = (np.hypot(*grad) > 0)
        xy_coords = self.getCoordinates(boundary)
        
        # reorder points using angle wrt center of image
        center = self._computeCenter()
        theta = np.arctan2(xy_coords[:, 1]-center[1], xy_coords[:,0]-center[0])
        boundary = np.append(xy_coords, theta[..., np.newaxis], 1)
        boundary = boundary[boundary[:, 2].argsort()]
        boundary = boundary[:, 0:-1]

        return boundary
    
    def getBoundary(self, method: str="opencv"):
        """Generate NumPy array with points on boundary. Points are ordered so as to draw a continuous loop if connected with lines.

        Args:
            method (str, optional): Which method to use to extract the boundary. Defaults to "opencv".

        Returns:
            np.ndarray: Ordered, scaled and translated boundary points.
        """
        if method=="opencv":
            return self._boundary_opencv()
        elif method=="numpy":
            return self._boundary_numpy()
        else:
            raise NotImplementedError
    

# ----------------------------------------------
# Meshio helpers
# ----------------------------------------------
def gmsh2meshio(mesh, cell_type: str, prune_z=False):
    """Extract `GMSH` mesh and return `meshio` mesh.

    Args:
        mesh: GMSH mesh.
        cell_type (str): Type of mesh cells.
        prune_z (bool, optional): Remove the z-component of the mesh to return a 2D mesh. Defaults to False.

    Returns:
        out_mesh: Converted meshio mesh object.
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:geometrical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh
