# SimpleShapes
Generation and annotation of datasets of simple shapes

This code allows to generate images and dataset of simple shapes, by inserting them with random positions and orientations in an empty image, with option to allow overlapping or not.
Several shapes are possible: basic geometric shapes (rectangle, triangle, ellipse), simple object shapes made from basic shapes (10 classes: house, tree, plane, jet, rocket, boat, submarine, tractor, car, truck), with random parameters, or shapes coming from a given dataset like Sharvit (see output samples).

<p float="left">
  <img src="https://github.com/RobinDelearde/SimpleShapes/blob/main/samples/SimpleShapes_samples.png" height="256" width="256" />
  <img src="https://github.com/RobinDelearde/SimpleShapes/blob/main/samples/4SimpleShapes_samples.png" height="256" width="256" />
  <img src="https://github.com/RobinDelearde/SimpleShapes/blob/main/samples/4SharvitShapes_samples.png" height="256" width="256" />
  <img src="https://github.com/RobinDelearde/SimpleShapes/blob/main/samples/2SimpleShapes_samples_sorted.png" height="256" width="256" />
</p>

This code was used to generate the dataset 2SimpleShapes used in the following papers:
```
@inproceedings{delearde_PR2021,
  author = {Deléarde, Robin and Kurtz, Camille and Wendling, Laurent},
  title = {Description and recognition of complex spatial configurations of object pairs with Force Banner 2D features},
  booktitle = {Pattern Recognition (PR)},
  publisher = {Elsevier},
  year = {2022},
  volume = {123},
  pages = {108410}
}

@inproceedings{delearde_ORASIS2021,
  title = {Description et reconnaissance de relations spatiales avec le bandeau de forces},
  author = {Deléarde, Robin and Kurtz, Camille and Dejean, Philippe and Wendling, Laurent},
  booktitle = {ORASIS},
  year = {2021}
}
```
