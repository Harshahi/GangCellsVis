# import scipy.io
# import napari

# # load .mat file
# mat = scipy.io.loadmat("bint_fishmovie32_100.mat")

# data = mat["bint"]   # shape: (297, 160, 950)

# viewer = napari.Viewer()
# viewer.add_image(data)

# napari.run()

import napari
import scipy.io

mat = scipy.io.loadmat("bint_fishmovie32_100.mat")
data = mat["bint"]

viewer = napari.Viewer()

viewer.add_image(
    data,
    contrast_limits=(0,1),
    colormap="magenta"
)

napari.run()