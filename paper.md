---
title: 'TopSearch: a Python package for topographical analysis of machine learning models and physical systems'
tags:
  - Python
  - machine learning
  - topography
  - energy landscapes
  - chemical physics
authors:
  - name: Luke Dicks
    orcid: 0000-0002-5278-4412
    affiliation: "1"
  - name: Edward O. Pyzer-Knapp
    corresponding: true
    orcid: 0000-0002-8232-8282
    affiliation: 1
affiliations:
 - name: IBM Research Europe, Hartree Centre, Daresbury, United Kingdom
   index: 1
date: 2 April 2024
bibliography: paper.bib
---

# Summary

Machine learning (ML) is now ubiquitous in all scientific fields, but there remains a significant challenge to understanding and explaining model performance [@Zhang2021; @Angelov2021]. Therefore, there is increasing interest in applying methods from other scientific disciplines (e.g. physics and biology) to improve the performance and explainability of machine learning algorithms [@Hassabis2017; @Karniadakis2021]. One methodology that has proved useful to understand machine learning performance is the energy landscape framework from chemical physics [@Wales:2003].

The energy landscape framework is a set of algorithms that map the topography of continuous surfaces by their stationary points. The topography is encoded as a weighted graph [@Noe2008] and in application to potential energy surfaces all physical properties of a system can be extracted from this graph [@Swinburne2020]. Examples of the methodology applied to potential energy surfaces explain physical phenomena for proteins [@Roder2019], small molecules [@Matysik2021], atomic clusters [@Csanyi2023] and crystalline solids [@Pracht2023].

Since the energy landscape framework is applicable to any given continuous surface, the methodology can also be applied to a wide range of machine learning algorithms through the corresponding loss function surface. Fitting of a machine learning model usually aims to locate low-valued or diverse solutions, and an understanding of the solution space topography explains model reproducibility and performance. Leveraging the energy landscape framework the performance and reliability of neural networks [@Niroomand2022], Gaussian processes [@Niroomand:2023] and clustering algorithms [@Dicks2022; @Dicks2023; @Wu2023] has been explored. Moreover, it has been used to explain the effect of dataset roughness on ML model performance [@Dicks2024]. A tutorial review of different applications is given in @Niroomand:2024.

# Statement of need

The `topsearch` Python package provides a rapid prototyping software for application of the energy landscape framework. It contains the functionality to be used for both potential energy surfaces and the loss function surfaces of varied machine learning models.

There is limited software for explicitly analysing the topography of loss function surfaces. These surfaces are considered implicitly when optimising an ML model through local minimisation, but none attempt to capture global topographical features of the parameter space. There is significantly more software for analysing potential energy surfaces, the majority of which approximate topographical features indirectly. Popular examples that aim to explore diverse regions of the surface through enhanced sampling are PyEMMA [@scherer_pyemma_2015] and large molecular simulation suites such as LAMMPS [@LAMMPS], GROMACS [@GROMACS], and AMBER [@AMBER] the simulations of which can be simplified using PLUMED [@PLUMED]. Explicit location of topographical features, such as stationary points, is more common in quantum chemistry and can be performed by software such as VTST [@vtst], PASTA [@Kundu2018], PyMCD [@Lee2023] and ORCA [@Neese2020]. The explicit computation of topography using the energy landscape framework has several advantages for application to machine learning and none of the above software contains all the required functionality.

Current leading tools for applying the energy landscape framework are the suite of FORTRAN programs: GMIN [@gmin] OPTIM [@optim] and PATHSAMPLE [@pathsample]. This software implements almost all functionality described within the energy landscape literature and, being written in a compiled language, is highly performant. Whilst a clear choice for production work where performance is critical, it is not without limitations for rapid prototyping. The user requires a detailed understanding of, and to pass information between, three large distinct pieces of software. There is a Python wrapper, `pylfl` [@pylfl], which simplifies their use, but does not remove the limitation of multiple programs that all require a detailed understanding. Furthermore, the software suite contains limited support for machine learning models, and addition of new models is challenging and time-consuming due to a lack of implementations of ML libraries in FORTRAN. Therefore, there is a need for a single software that performs the energy landscape framework for both ML and physics, which integrates seamlessly with ML libraries, thus enabling rapid prototyping in this domain.

`topsearch` replaces the functionality of the FORTRAN software suite in a single software package, reducing the need for data transfer and subsequent parameterisation and setup. The package, written entirely in Python, contains additional novel functionality for machine learning, and due to the prevalence of Python in machine learning further new models can be included quickly and easily. Furthermore, the implementation is significantly shorter, containing less than a hundredth of the lines of code; enabling faster developer onboarding.

# Applications

The Github repository ([https://github.com/IBM/topography-searcher](https://github.com/IBM/topography-searcher)) contains examples for varied applications, which are listed in turn below.

- `example_function` - This folder contains examples for mapping the surface topography of an arbitrary function. The examples provide an introduction to the methodology, and illustrate the major code functionality. Application to two-dimensional functions allows direct visualisation of the surfaces, which makes clear the topographical analysis.
- `dataset_roughness` - Illustration of the novel code application to quantify dataset roughness [@Dicks2024]. This analysis can uniquely explain and predict ML regression performance both globally and locally, even in the absence of training data.

Atomic and molecular systems require significant additional functionality. However, the examples illustrate that the scripts remain remarkably similar.

- `atomic` - An example that performs exploration of the potential energy surface of a small atomic cluster. 

- `molecular` - These examples illustrate how to explore the potential energy surfaces of small molecules using quantum chemistry.

This list of examples does not form an exhaustive set of use cases. Previous applications of this methodology, which will also be possible using `topsearch`, are protein and nucleic acids potential energy surfaces and Gaussian process, neural network and clustering loss function surfaces. Moreover, there are many additional machine learning models that could be analysed, and the Python implementation allows for their rapid inclusion.

# Conclusions

The `topsearch` Python package fulfils the need for a rapid prototyping and analysis tool for the energy landscape framework that can be applied to both physics and machine learning models. This software is significantly more lightweight than existing solutions; a large reduction in code and integration in a single piece of software ensures the Python implementation is significantly easier to develop. Moreover, the package provides a simpler interface for accessing the functionality, and in tandem with detailed examples, results in a shallower learning curve for use within diverse applications. Lastly, the software is unique in the amount of machine learning models that can be explored and and can easily be extended with existing Python implementations. Our aim is that this software package will aid diverse researchers from computer science to chemistry by providing a simple solution for application of the energy landscape framework.

# Acknowledgements

LD and EOP-K would like to acknowledge the financial support of the Hartree National Centre for Digital Innovation – a collaboration between the Science and Technology Facilities Council and IBM. The authors would also like to thank Nicholas Williams, Matthew Wilson, Nicolas Galichet and Vlad C\u{a}rare for their helpful feedback as early users of the package.

# References
