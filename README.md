# Attentional Ptycho-Tomography (APT)
This is the training code for the deep learning model called Attentional Ptycho-Tomography (APT) for three-dimensional nanoscale X-ray imaging. We aimed for minimizing data acquisition and computation time using machine learning. The paper is originally published in <a href="https://www.nature.com/articles/s41377-023-01181-8">Light: Science and Applications</a>, and the arXiv paper is available <a href="https://arxiv.org/abs/2212.00014">here</a>.

## **Abstract**
Noninvasive X-ray imaging of nanoscale three-dimensional objects, such as integrated circuits (ICs), generally requires two types of scanning: ptychographic, which is translational and returns estimates of the complex electromagnetic field through the IC; combined with a tomographic scan, which collects these complex field projections from multiple angles. Here, we present Attentional Ptycho-Tomography (APT), an approach to drastically reduce the amount of angular scanning, and thus the total acquisition time. APT is machine learning-based, utilizing axial self-Attention for Ptycho-Tomographic reconstruction. APT is trained to obtain accurate reconstructions of the ICs, despite the incompleteness of the measurements. The training process includes regularizing priors in the form of typical patterns found in IC interiors, and the physics of X-ray propagation through the IC. We show that APT with ×12 reduced angles achieves fidelity comparable to the gold standard Simultaneous Algebraic Reconstruction Technique (SART) with the original set of angles. When using the same set of reduced angles, then APT also outperforms Filtered Back Projection (FBP), Simultaneous Iterative Reconstruction Technique (SIRT) and SART. The time needed to compute the reconstruction is also reduced, because the trained neural network is a forward operation, unlike the iterative nature of these alternatives. Our experiments show that, without loss in quality, for a 4.48 × 93.2 × 3.92 µm3 IC (≃ 6 × 10^8 voxels), APT reduces the total data acquisition and computation time from 67.96 h to 38 min. We expect our physics-assisted and attention-utilizing machine learning framework to be applicable to other branches of nanoscale imaging, including materials science and biological imaging.

## Citation
If you find the paper useful in your research, please consider citing the paper:


	@article{kang_attentional_2023,
		title = {Attentional {Ptycho}-{Tomography} ({APT}) for three-dimensional nanoscale {X}-ray imaging with minimal data acquisition and computation time},
		volume = {12},
		issn = {2047-7538},
		url = {https://doi.org/10.1038/s41377-023-01181-8},
		doi = {10.1038/s41377-023-01181-8},
		number = {1},
		journal = {Light: Science \& Applications},
		author = {Kang, Iksung and Wu, Ziling and Jiang, Yi and Yao, Yudong and Deng, Junjing and Klug, Jeffrey and Vogt, Stefan and Barbastathis, George},
		month = may,
		year = {2023},
		pages = {131},
	}

