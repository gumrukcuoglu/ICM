# SDM notebooks

The repo accompanies the paper [arXiv:2404.16658](https://arxiv.org/abs/2404.16658) which presents an inference framework based on a spherical diffusion model (SDM). This repository contains example notebooks demonstrating inference from SDM using synthetic data generated via a pseudo-2D (P2D) battery model. A comparative analysis with the traditional _Galvanostatic Intermittent Titration Technique_ (GITT) is also included.

Simulations based on the P2D model are performed using [Dandeliion](http://www.dandeliion.com). Additional code for estimating equilibrium potentials from pseudo-OCV (pOCV) data—also derived from simulations—will be added shortly.

## Requirements
`scipy`, `matplotlib`, `numpy`.

## Usage
Examples will be placed in the Examples/ folder at some point.

## Version history:

- v0.1 – Initial release containing a preliminary, unorganised set of notebooks.

## TO DO:

- Finalise the example with GITT, then a comparison with the ISDM approach. Basically, clean up the package to be made public.
- The above should be made flexible since I also plan to use real data to re-generate the plots in the paper. This part will not be public.

