---
layout: external
title: Scaling up Bayesian variational inference using distributed computing clusters
role: International Journal of Approximate Reasoning, 2017
category: papers
external_url:
tags: [sade, rnb, smooth]
image:
  thumb: pgm2016logo.png
published: true
---

**First Author**. Joint work with Ana M Martinez, Helge Langseth, Thomas D Nielsen, Antonio Salmerón, Darío Ramos-López, Anders L Madsen.


<!--

In this paper we present an approach for scaling up Bayesian learning using variational
methods by exploiting distributed computing clusters managed by modern big data processing
tools like Apache Spark or Apache Flink, which efficiently support iterative map-reduce
operations. Our approach is defined as a distributed projected natural gradient ascent
algorithm, has excellent convergence properties, and covers a wide range of conjugate
exponential family models. We evaluate the proposed algorithm on three real-world datasets
from different domains (the Pubmed abstracts dataset, a GPS trajectory dataset, and a
financial dataset) and using several models (LDA, factor analysis, mixture of Gaussians
and linear regression models). Our approach compares favorably to stochastic variational
inference and streaming variational Bayes, two of the main current proposals for scaling
up variational methods. For the scalability analysis, we evaluate our approach over a
network with more than one billion nodes and approx.  latent variables using a computer
cluster with 128 processing units (AWS). The proposed methods are released as part of
an open-source toolbox for scalable probabilistic machine learning [http://www.amidsttoolbox.com](http://www.amidsttoolbox.com).


Andrés R Masegosa, Ana M Martinez, Helge Langseth, Thomas D Nielsen, Antonio Salmerón, Darío Ramos-López, Anders L Madsen.
Scaling up Bayesian variational inference using distributed computing clusters. International Journal of Approximate Reasoning, 88, 435-451. 2017.
-->
<a href="https://www.sciencedirect.com/science/article/pii/S0888613X17303985"><i class="fa fa-file-pdf-o" aria-hidden="true"> PDF</i></a> <a href="https://github.com/amidst/toolbox"><i class="fa fa-github" aria-hidden="true" > Github</i></a> <a href="/papers/PGM2016-slides.pdf"><i class="fa fa-line-chart" aria-hidden="true" > Slides</i></a>
