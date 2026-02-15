# Hierarchical Spiking Neural Network

This project models the primate ventral visual system (PVVS) as a hierarchical spiking neural network (HSNN) trained via spike-timing-dependent plasticity (STDP). This is to study the emergence of feature selectivity of shapes and to evaluate a hypothesis for feature binding - hierarchical binding by polychronization<sup>[1,2]</sup>.

## Features
- Biologically inspired model of the primate ventral visual system (PVVS)
- Competitive unsupervised learning via STDP
- Brian2-based spiking neural network simulation<sup>[3]</sup>
- Analysis of shape feature selectivity and hierarchical binding by polychronization

## Installation

This project requires [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to be installed.
Developed on Ubuntu 22.04 LTS.

To install the `hsnn` library:
1. Clone the repository and navigate to the root directory:
    - `git clone https://github.com/BCGardner/feature_binding.git`
    - `cd feature_binding`
2. Create a new `conda` environment called `hsnn` and install the required dependencies:
    - `conda env create -f environment.yaml`

## Large File Storage (Git LFS)

Some large files (such as model checkpoints) are tracked using [Git Large File Storage (LFS)](https://git-lfs.github.com/).

> **Note:** Downloading all LFS-tracked files will require approximately 7 GB of disk space.

After cloning the repository, run the following command to download these files:

```bash
git lfs pull
```

If you do not have Git LFS installed, you can install it on Linux with:

```bash
sudo apt-get install git-lfs
git lfs install
```

This ensures all required large files are available for running experiments and notebooks.

> **Troubleshooting:**
> If you see an error like
> `tls: failed to verify certificate: x509: certificate signed by unknown authority`
> when running `git lfs pull`, your system may not trust the server's TLS certificate.
> As a temporary workaround, you can disable SSL verification for Git LFS:
>
> ```bash
> git config http.sslVerify false
> ```
>
> Then run `git lfs pull` again.

## Usage

Before running project scripts, ensure they have execution permissions. You can grant permission to all scripts in the scripts/ directory with the following command:
```bash
find scripts/ -name "*.py" -exec chmod +x {} +
```

To regenerate artifacts required to run notebooks (e.g. for `notebooks/figures/plot_fig5.ipynb`):
```bash
# Generates spike recording artifacts.

# 1) Before network training
./scripts/run_main_workflow.py experiments/n3p2/train_n3p2_lrate_0_04_181023 31 --rule inference -v

# 2) After network training (include checkpoint)
./scripts/run_main_workflow.py experiments/n3p2/train_n3p2_lrate_0_04_181023 31 --chkpt -1 --rule inference -v
```
To regenerate all artifacts, including spike recordings, PNG detections, significance testing:
```bash
# Alternatively include `--rule significance`
./scripts/run_main_workflow.py experiments/n3p2/train_n3p2_lrate_0_04_181023 31 --chkpt -1 -v
```

## Troubleshooting

**Ray on Ubuntu 24.04+**
If you encounter errors related to the Ray framework, please note that the current version installed in this project currently has known issues on Ubuntu versions exceeding 22.04 LTS. This will not interfere with the artifact regeneration scripts provided in the Usage section.

## Authors

Codebase developed by Brian Gardner. Manuscript authors are as follows:

Brian Gardner<sup>1,*</sup>,
Patrick T. McCarthy<sup>2</sup>,
Joseph Chrol-Cannon,
Dan F. M. Goodman<sup>3</sup>,
Simon R. Schultz<sup>4</sup>,
Giovanni Lo Iacono<sup>1,5</sup>,
Simon M. Stringer<sup>6</sup>

<sup>1</sup> <sub>Department of Comparative Biomedical Sciences, School of Veterinary Medicine, University of Surrey, Guildford, United Kingdom</sub>\
<sup>2</sup> <sub>Centre for Neural Circuits and Behaviour, University of Oxford, Oxford, United Kingdom</sub>\
<sup>3</sup> <sub>Department of Electrical and Electronic Engineering, Imperial College London, South Kensington, London, United Kingdom</sub>\
<sup>4</sup> <sub>Department of Bioengineering, Imperial College London, South Kensington, London, United Kingdom</sub>\
<sup>5</sup> <sub>The Surrey Institute for People-Centred AI, University of Surrey, Guilford, United Kingdom</sub>\
<sup>6</sup> <sub>Centre for Theoretical Neuroscience and Artificial Intelligence, Department of Experimental Psychology, University of Oxford, Oxford, United Kingdom</sub>

<sup>*</sup>  b.gardner@surrey.ac.uk

## References

[1] [Eguchi, A., Isbister, J. B., Ahmad, N., & Stringer, S. (2018). The emergence of polychronization and feature binding in a spiking neural network model of the primate ventral visual system. Psychological review, 125(4), 545.](https://www.oftnai.org/articles/Brain_modelling_articles/Publications/Vision/2018-25960-001.pdf)\
[2] [Izhikevich, E. M. (2006). Polychronization: computation with spikes. Neural computation, 18(2), 245-282.](https://www.mitpressjournals.org/doi/pdfplus/10.1162/089976606775093882)\
[3] [https://brian2.readthedocs.io/en/stable/index.html](https://brian2.readthedocs.io/en/stable/index.html)

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

The AGPL-3.0 is a strong copyleft license specifically designed for software that may be run over a network. It ensures that if you modify the software and provide its functionality to others over a network (like a web service), you must also share your modified source code.

See the [LICENSE](LICENSE) file for the full license text.
