import os
from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence, SMPLLayer

# Set model path
os.environ["SMPLX_MODEL_DIR"] = "/home/danielamrhein/master_thesis/data/smplx_models"

# Use PyQt5 window instead of Pyglet
C.window_type = "pyqt5"

if __name__ == "__main__":
    smpl_layer = SMPLLayer(
        model_type="smplx",
        gender="neutral",  # Make sure SMPLX has a neutral model
    )

    v = Viewer()
    v.scene.add(SMPLSequence.t_pose(smpl_layer))
    v.run()
