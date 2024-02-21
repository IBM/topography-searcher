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
date: 21 February 2024
bibliography: paper.bib
---

# Summary

Machine learning (ML) is now ubiquitous in all scientific fields, but there remains a significant challenge to understanding and explaining model performance.\cite{Zhang2021} Therefore, there is increasing interest in applying methods from other scientific disciplines (e.g. physics and biology) to improve the performance and explainability of machine learning algorithms.\cite{Karniadakis2021} One methodology that has proved useful to understand machine learning performance is the energy landscape framework from chemical physics.\cite{Wales2003}

The energy landscape framework is a set of algorithms that map the topography of continuous surfaces by their stationary points. The topography is encoded as a weighted graph\cite{Noe2008} and in application to potential energy surfaces all physical properties of a system can be extracted from this graph.\cite{Swinburne2020} Examples of the methodology applied to potential energy surfaces explain physical phenomena for proteins,\cite{Roder2019} small molecules,\cite{Matysik2021} atomic clusters\cite{Csanyi2023} and crystalline solids.\cite{Pracht2023}

Additionally, the energy landscape framework is applicable to any given continuous surface, allowing application to a wide range of machine learning algorithms through the corresponding loss function surface. Fitting of a machine learning model usually aims to locate low-valued or diverse solutions, and an understanding of the solution space topography explains model reproducibility and performance. Leveraging the energy landscape framework the performance and reliability of neural networks,\cite{Niroomand2022} Gaussian processes\cite{Niroomand2023} and clustering algorithms\cite{Dicks2022, Dicks2023, Wu2023} has been explored. Moreover, it has been used to explain the effect of dataset roughness on ML model performance,\cite{Dicks2023_2} and a tutorial review of different applications is given in \cite{Niroomand2024}.

# Statement of need

The \verb +topsearch+ Python package provides a rapid prototyping software for application of the energy landscape framework. It contains the functionality to be used for both potential energy surfaces and the loss function surfaces for a variety of machine learning models. We describe alternate software for both loss function and potential energy surface exploration in the following paragraphs.

There is limited software for explicitly analysing the topography of loss function surfaces. These surfaces are considered implicitly when optimising an ML model through local minimisation, but none attempt to capture global topographical features of the parameter space. There is significantly more software for analysing potential energy surfaces, the majority of which approximate topographical features indirectly. Popular examples that aim to explore diverse regions of the surface through enhanced sampling are PyEMMA\cite{scherer_pyemma_2015} and large molecular simulation suites such as LAMMPS,\cite{LAMMPS} GROMACS,\cite{GROMACS} and AMBER\cite{AMBER} the simulations of which can be simplified using PLUMED.\cite{PLUMED} Explicit location of topographical features, such as stationary points, is more common in quantum chemistry and can be performed by software such as VTST,\cite{vtst} PASTA,\cite{Kundu2018} PyMCD\cite{Lee2023} and ORCA.\cite{Neese2020} The explicit computation of topography using the energy landscape framework has several advantages for application to machine learning and none of the above software contains all the required functionality.

Current leading tools for applying the energy landscape framework are the suite of FORTRAN programs: GMIN,\cite{GMIN} OPTIM\cite{OPTIM} and PATHSAMPLE.\cite{PATHSAMPLE} This software implements almost all functionality described within the energy landscape literature and, being written in a compiled language, is highly performant. Whilst a clear choice for production work where performance is critical, it is not without limitations for rapid prototyping. The user is required to be familiar with, and pass information between, three large distinct pieces of software. There exists a Python wrapper, \verb +pylfl+,\cite{pylfl} which simplifies their use, but does not remove the limit of multiple programs each of which requires a detailed understanding for use. Furthermore, the software suite contains limited support for machine learning models, and addition of new models is challenging and time-consuming due to a lack of implementations of ML libraries in FORTRAN. Therefore, there is a need for a single software that performs the energy landscape framework for both ML and physics, which integrates seamlessly with ML libraries, thus enabling rapid prototyping in this domain.

\verb +topsearch+ replaces the functionality of the FORTRAN software suite in a single software package, reducing the need for data transfer and subsequent parameterisation and setup. The package, written entirely in Python, contains additional novel functionality for machine learning, and due to the prevalence of Python in machine learning further new models can be included quickly and easily. Furthermore, the implementation is significantly shorter, requiring less than a hundredth of the lines of code; enabling faster developer onboarding.

# Applications

The Github repository, located at \url{https://github.com/IBM/topography-searcher}, contains examples for varied applications. Initially, we provide the examples for mapping surfaces of simple test functions to illustrate the major functionality of the code. These examples constitute an introduction to the methodology and its outputs, which aids the understanding of both the theory and code implementation. There are additional examples for its use in machine learning with an application to quantifying dataset roughness `[@Dicks2023_2]`. This novel application, only possible with this software, can uniquely explain and predict regression performance without any model training. We also provide examples for both atomic and molecular systems, which require significant additional functionality. However, the examples illustrate that the scripts remain remarkably similar, leading to a shallow learning curve because a small number of methods work in all applications.

This list of examples does not form the complete set of use cases. Previous applications of this methodology, which will be additionally possible using \verb +topsearch+, are protein and nucleic acids potential energy surfaces and Gaussian process, neural network and clustering loss function surfaces. Moreover, there are many additional machine learning models that could be analysed and the Python implementation allows for their rapid inclusion.

# Conclusions

The \verb +topsearch+ Python package fills a need for a rapid prototyping and analysis tool for the energy landscape framework that can be applied to both physics and machine learning models. This software solution is significantly more lightweight than existing programs and provides a simpler interface for accessing the functionality, all within a single piece of software. Therefore, with a great reduction in code the Python implementation is significantly easier to develop. The learning curve for use is also shallow, and we provide detailed examples that illustrate the conservation of scripts between diverse applications. Lastly, the software is unique in the amount of machine learning models that can be explored and and can easily be extended with existing Python implementations. Our aim is that this software package will aid diverse researchers from computer science to chemistry by providing a simple solution for application of the energy landscape framework.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Acknowledgements

LD and EOP-K would like to acknowledge the financial support of the Hartree National Centre for Digital Innovation – a collaboration between the Science and Technology Facilities Council and IBM. The authors would also like to thank Nicholas Williams and Vlad C\v{a}rare for their helpful feedback as early users of the package.

# References
