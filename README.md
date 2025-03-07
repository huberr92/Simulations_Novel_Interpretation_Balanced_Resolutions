# A Novel Interpretation of the Radon Transform’s Ray- and Pixel-Driven Discretizations under Balanced Resolutions
[![ArXiv Paper](https://arxiv.org/abs/2501.11451)](https://arxiv.org/abs/2501.11451)
[![SSVM Conference 2025](https://sites.google.com/view/ssvm-2025/home-page)](https://sites.google.com/view/ssvm-2025/home-page)

This repository accompanies the paper ** A Novel Interpretation of the Radon Transform’s Ray- and Pixel-Driven Discretizations under Balanced Resolutions ** authored by Dr. Richard Huber 
published at the SSVM 2025 (10th International Conference on Scale Space and Variational Methods in Computer Vision).
It contains the code that was used in the simulations and numerical. It contains 4 individual Python files that contain the code for independent numerical experiments.
All of the use the gratopy_expo module also provided in this data set. This is an updated version of the 
[![Gratopy](https://gratopy.readthedocs.io/en/latest/)](https://gratopy.readthedocs.io/en/latest/) module which contains ray-driven implementations that are not yet published in the main gratopy branch on GitHub.
The implementation uses parallelization via [![PyOpenCL](https://documen.tician.de/pyopencl/)](https://documen.tician.de/pyopencl/), which allows for fast execution, but dependent on the hardware used can be a bit tedeaous to install/set up.

## Content
* **Example1.py** contains the experiments related to the Example1 section of the paper, more precisely Figure 4. It executes the Ray-driven and Pixel-driven Radon transforms of the characteristic function of an ellipse.
 This is repeated for balanced resolution settings with increasing Nx=Ns starting at 200 and going up to 4000 in 11 steps. Moreover, we execute this for 180 or 360 projections. We plot the projections for one specific setting Nx=Ns=580 and N\phi=360, as well as
 the evolution of errors with increasing levels of discretization.
 
* **Example2_part1.py** contains the experiments related to the Example2 section of the paper, more precisely Figure 5. It executes the Ray-driven and Pixel-driven backprojections for a sinogram that is constantly one.
 This is executed for 3 different settings, the balanced resolutions Nx=Ns=1000, Nx=Ns=4000 and the unbalanced setting Nx=1000, Ns=4000.

 * **Example2_part2.py** contains the experiments related to the Example2 section of the paper, more precisely Figure 6. It executes the Ray-driven for a sinogram that is constantly one as well as of a sinogram with only one active angle, all others being zero.
 This is executed for increasing number of angles from 30 to 720 in 47 steps, in balanced resolutions Nx=Ns=2000. The evolution of the corresponding approximation errors is plotted.
 
 * **Example3.py** contains the experiments related to the Example3 section of the paper, more precisely Figure 7. It executes the Ray-driven and Pixel-driven backprojections for a sinogram that is linear in the detector but constant in the angular dimension.
 This is executed for  balanced resolutions Nx=Ns=4000 and fixed number of angles 360, where for the pixel-driven method we once correct angles by a shift by half a degree. 
 

 </table>

## Installation

No dedicated installation is needed for this software, the code can simply be downloaded and executed when importing the contents of the `gratopy_expo` directory as a module. Please make sure to have the following Python modules installed, most of which should be standard.

## Requirements

* [pyopencl>=2019.1](https://pypi.org/project/pyopencl/)
* [numpy>=1.17.0](https://pypi.org/project/numpy/)
* [scipy>=1.3.0](https://pypi.org/project/scipy/)
* [matplotlib>=3.2.0](https://pypi.org/project/matplotlib/)

Note that in particular, correctly installing and configuring PyOpenCL might take some time, as dependent on the used platform/GPU, suitable drivers must be installed. We refer to [PyOpenCL's documentation](https://documen.tician.de/pyopencl/).


## Getting started
The examples can be executed in a bash using ``python Example1.py'' or in a python environment using ``run Example1.py''.

When run the examples, after a few secconds you will be asked by PyOpenCL to ``choose a platform''. The options depend on your hardware (and potentially the drivers used), where
``Portable Computing Language'' is the standard option for parallel execution on the CPU, but if you have the possibility, GPU executions will probably be faster.


## Authors

* **Richard Huber**, Technical University of Graz, richu@dtu.dk, http://richardhuber-math.com/.

The author is affiliated with the [Department of Applied Mathematics and Computer Science](https://orbit.dtu.dk/en/organisations/department-of-applied-mathematics-and-computer-science) at the [Technical University of Denmark (DTU)](https://www.dtu.dk/english/).

## Publications
If you find this tool useful, please cite the following associated publication.

* Richard Huber: A novel interpretation of the Radon transform’s ray- and pixel-driven discretizations under balanced resolutions. Accepted to the SSVM 2025 (Scale Space and Variational Methods in Computer Vision) (2025)
* Kristian Bredies and Richard Huber. (2021). *Gratopy 0.1* [Software]. Zenodo. https://doi.org/10.5281/zenodo.5221442

## Acknowledgements

The development of this software was supported by the following project

* *The Villum Foundation* (Grant No.25893)

This code uses the Gratopy toolbox as a base, whose development was supported by 

* *Regularization Graphs for Variational Imaging*, funded by the Austrian Science Fund (FWF), grant P-29192,

* *International Research Training Group IGDK 1754 Optimization and Numerical Analysis for Partial Differential Equations with Nonsmooth
Structures*, funded by the German Research Council (DFG) and the Austrian Science Fund (FWF), grant W-1244.


## License

This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
