# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2021-04-10 12:18:17
# @Last Modified by:   jsgounot
# @Last Modified time: 2021-04-10 15:26:16

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import Normalize

class Externals :
    
    def __init__(self, offset=0, pad=0, size=None) :
        self.size = size
        self.pad = pad
        self.offset = offset
        self.initial_depth = None

    def set_initial_depth(self, value) :
        self.initial_depth = value

    def set_size(self, mdepth) :
        self.size = self.size or mdepth * .2

    def fsize(self) :
        return self.size + self.offset

    @property
    def depth(self):
        return self.initial_depth + self.offset + self.pad
    
    def draw(self, ax, angles, tree, arc, start) :
        pass

class Externals_background(Externals) :
    
    def __init__(self, bg_internal=False, ** kwargs) :
        super().__init__(** kwargs)
        self.bg_internal = bg_internal

    @staticmethod
    def get_wedge_angles(clade, angles) :
        min_angle = min(angles[child]
            for child in clade.get_terminals())
        
        max_angle = max(angles[child]
            for child in clade.get_terminals())

        nnodes = len(clade.get_terminals())
        angles_diff = ((max_angle - min_angle) / (nnodes + 2)) / 2
        
        min_angle = min_angle - angles_diff
        max_angle = max_angle + angles_diff

        return min_angle, max_angle

    def from_internal(self, ax, angles, tree) :
        depth = self.initial_depth + self.offset + self.size

        for clade in tree.get_nonterminals() :
            try : wedge = clade.wedge
            except AttributeError : continue

            min_angle, max_angle = Externals_background.get_wedge_angles(clade, angles)

            yield Wedge((0, 0), depth, min_angle, max_angle, width=self.size,
                transform=ax.transData._b, ** clade.wedge)


    def draw(self, ax, angles, tree, arc, start) :
        if self.bg_internal :
            wedges = self.from_internal(ax, angles, tree)

        else :
            wedges = []

        for wedge in wedges :
            ax.add_patch(wedge) 

        super().draw(ax, angles, tree, arc, start)

    def make_background(self, ax) :
        pass

        """
        wedge = Wedge((0, 0), mdepth + lsize, min_angle, max_angle, width=lsize,
            transform=ax.transData._b, ** kwargs)
        # ha = "right" if 270 > min_angle > 90 else "left"  
        """

class Externals_labels(Externals_background) :
    
    def __init__(self, data=None, ** kwargs) :
        super().__init__(** kwargs)
        self.data = data

    def draw(self, ax, angles, tree, arc, start) :

        # first we draw parent classes
        super().draw(ax, angles, tree, arc, start)

        depth = self.depth
        data = self.data or {leaf : leaf.name for leaf in tree.get_terminals()}

        for leaf, label in data.items() :
            angle = angles[leaf]
            rotation = angle + 180 if 270 > angle > 90 else angle
            ha = "right" if 270 > angle > 90 else "left"
            rad = np.deg2rad(angle)    
            ax.text(rad, depth, label, rotation=rotation, ha=ha, va="center", rotation_mode='anchor')

class Externals_patchs(Externals_background) :
    
    def __init__(self, data=None, ** kwargs) :
        super().__init__(** kwargs)
        self.data = data

    def draw(self, ax, angles, tree, arc, start) :

        # first we draw parent classes
        super().draw(ax, angles, tree, arc, start)

        depth = self.depth + (self.size / 2)
        data = self.data

        if not data :
            data = {}
            for leaf in tree.get_terminals() :
                try : data[leaf] = leaf.patch
                except AttributeError : continue

        for leaf, patch in data.items() :

            angle = angles[leaf]
            rad = np.deg2rad(angle)
            x_coor = depth * np.cos(rad)
            y_coor = (depth * np.sin(rad))
            
            patch = patch((x_coor, y_coor))
            patch.set_transform(ax.transData._b)
            patch.set_zorder(3)
            ax.add_patch(patch)

class Externals_heatmap(Externals) :
    
    def __init__(self, data, cmap=None, row_order=None,
            wedges_kwargs={}, label=True, ** kwargs) :
        
        super().__init__(** kwargs)
        self.data = data

        generator = (value for linfo in data.values() for value in linfo.values())
        min_value = min(generator)
        generator = (value for linfo in data.values() for value in linfo.values())
        max_value = max(generator)

        print (min_value, max_value)

        cmap = cmap or plt.get_cmap("afmhot")
        self.cmap = cmap
        self.norm = Normalize(vmin=min_value, vmax=max_value)

        self.row_order = row_order or sorted({
            rowname for linfo in data.values()
            for rowname in linfo.keys()
            })

        self.wedges_kwargs = wedges_kwargs
        self.label = label

    def draw(self, ax, angles, tree, arc, start) :

        # first we draw parent classes
        super().draw(ax, angles, tree, arc, start)

        wedges_kwargs = {
            "edgecolor" : "black",
            "linewidth": 0,
            "antialiased": True
        }

        wedges_kwargs.update(self.wedges_kwargs)

        hdepth = self.depth
        vertical_cell_size = self.size / len(self.row_order)
        horizontal_cell_size = (arc / len(tree.get_terminals())) * .5

        for leaf, values in self.data.items() :
            
            leaf_angle = angles[leaf]
            min_angle = leaf_angle - horizontal_cell_size
            max_angle = leaf_angle + horizontal_cell_size

            for idx, row in enumerate(self.row_order) :
                value = values.get(row, np.nan)

                depth = hdepth + (vertical_cell_size * idx) + vertical_cell_size * .5
                color = self.cmap(self.norm(value))

                print (leaf, row, value, min_angle, max_angle, depth, vertical_cell_size, color)

                wedge = Wedge((0, 0), depth, min_angle, max_angle, width=vertical_cell_size,
                    transform=ax.transData._b, color=color, ** wedges_kwargs)

                ax.add_patch(wedge)

        if self.label :
            angle = start - 90
            for idx, row in enumerate(self.row_order) :
                rad = np.deg2rad(start - horizontal_cell_size * 1.5)
                depth = hdepth + (vertical_cell_size * idx) - (vertical_cell_size * .5)
                ax.text(rad, depth, row, rotation=angle, ha="left", va="center", rotation_mode='anchor')