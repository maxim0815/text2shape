import os
from mpl_toolkits.mplot3d import Axes3D
import nrrd
import numpy as np
import matplotlib.pyplot as plt

class RenderImage():
    def __init__(self, image_dir=0):
        if image_dir != 0:
            self.img_name = image_dir[image_dir.rfind("/")+1:image_dir.rfind(".")]
            self.data, _ = nrrd.read(image_dir, index_order='C')            
        self.colors = []
        self.edge_colors = []

    def set_shape(self, shape):
        self.data = shape
    
    def set_name(self, name):
        self.img_name = name

    def _generate_colors(self, row):
        if row[3] >0:
            self.colors.append('#{:x}{:x}{:x}{:x}'.format(row[0], row[1], row[2], row[3]))
            self.edge_colors.append('#{:x}{:x}{:x}'.format(row[0], row[1], row[2]))
        else:
            self.colors.append('')
            self.edge_colors.append('')

    def render_voxels(self, datadir='renders'):
        np.apply_along_axis( self._generate_colors, axis=3, arr=self.data )
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        data_crop = self.data[:,:,:,0]
        colors_reshape = np.reshape(self.colors, data_crop.shape)
        edges_reshape = np.reshape(self.edge_colors, data_crop.shape)
        # TODO: marked as issue
        try:
            ax.voxels(data_crop, facecolors=colors_reshape, edgecolors=edges_reshape)
        except Exception as e:
            print(e)
            pass
        # save file
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        save_dir = os.path.join(datadir, self.img_name+".png")
        plt.savefig(save_dir)
        plt.close()


if __name__ == '__main__':
    img = RenderImage('nrrd_256_filter_div_32_solid/43321568c4bc0a7cbaf2e78ed413860a/43321568c4bc0a7cbaf2e78ed413860a.nrrd')
    img.render_voxels()

