---
layout: post-light-feature
title: Deep Probabilistic Modeling (IV). Probabilistic models with deep neural networks.
description: How deep neural networks can be used to extend the modelling capacities of a probabilistic model. 
category: articles
tags: [probabilistic models, probability models, exponential families, variational inference, autoencoder, neural networks, graphical models, inference, deep learning, machine learning]
image:
  feature: deeplvmfig.png
  thumb: deeplvmfig.png
published: true
---

[comment]: <> (How new developments in variational inference and probabilistic programming languages are given rise a new golden era for probabilistic modeling. )

<head>
  <meta charset="utf-8" />
  <meta name="generator" content="pandoc" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
  <title>models</title>
  <style type="text/css">
      code{white-space: pre-wrap;}
      span.smallcaps{font-variant: small-caps;}
      span.underline{text-decoration: underline;}
      div.column{display: inline-block; vertical-align: top; width: 50%;}
  </style>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
  <!--[if lt IE 9]>
    <script src="//cdnjs.cloudflare.com/ajax/libs/html5shiv/3.7.3/html5shiv-printshiv.min.js"></script>
  <![endif]-->
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({ 
    showProcessingMessages: false,
    messageStyle: 'none',
    'HTML-CSS': {
      imageFont: null
    },
    TeX: { 
    	equationNumbers: {autoNumber: "all"},     
    Macros: {
      bmy: "{\\mathbf{y}}",
      bmw: "{\\mathbf{w}}",
      bmh: "{\\mathbf{h}}",
      bmmu: "{\\mathbf{\\mu}}",
      bmx: "{\\mathbf{x}}",
      bmz: "{\\mathbf{z}}",
      bmbeta: "{\\mathbf{\\beta}}",
      bmalpha: "{\\mathbf{\\alpha}}",
      bmt: "{\\mathbf{t}}",
      bmI: "{\\mathbf{I}}",
      calN: "{\\mathcal{N}}",
      given: "{|}",
      bmlambda: "{\\mathbf{\\lambda}}",
      bmphi: "{\\mathbf{\\phi}}",
      E: "{\\mathbb{E}}",
      KL: "{KL}",
      lb: "{\\mathcal{L}}",
    } },     
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    }, });
  </script>
</head>

Deep Latent Variable Models {#sec:deepLVMs}
---------------------------

LVMs has been usually restricted to the conjugate exponential family
because, in this case, the inference was feasible (and scalable). But
recent advances in variational inference are inspiring many recent works extending
LVMs with DNNs. Variational Auto-encoders (VAE)
[@kingma2013auto; @doersch2016tutorial] are probably the most
influential work combining LVMs and deep neural networks. VAEs extend
the classic technique of principal component analysis (PCA)
[@jolliffe2011principal] for data representation in lower-dimensional
spaces. More precisely, [@kingma2013auto] extends the probabilistic
version of the PCA model [@tipping1999probabilistic] where the
relationship between the low-dimensional representation and the observed
data is governed by a DNN, i.e. a highly non-linear function, as opposed
to the standard linear assumptions of the basic version of the PCA
model. These new models were able to capture much more compact
low-dimensional representation especially in cases where data were
highly-dimensional and complex[^1]. As it is the case of image data
[@kingma2013auto; @kulkarni2015deep; @gregor2015draw; @sohn2015learning; @pu2016variational],
text data [@semeniuta2017hybrid], audio data [@hsu2017learning],
chemical molecules [@gomez2018automatic], to name some of the most
representative applications of this technique.

As commented above, VAEs have given rise to a plethora of extensions of
classic LVMs to their *deep countepart*. [@johnson2016composing]
contains different examples of this approach and proposes extensions of
Gaussian mixture models, latent linear dynamical systems and latent
switching linear dynamical systems with non-linear relationships
modelled by DNNs. [@linderman2016recurrent] extends hidden semi-Markov
models with recurrent neural networks.
[@zhou2015poisson; @card2017neural] extends popular LDA models
[@blei2003latent] for uncovering topics in text data. Many other works
are following this same trend
[@chung2015recurrent; @jiang2016variational; @xie2016unsupervised; @louizos2017causal].

LVMs with DNNs can also be found in the literature under the name of
deep generative models
[@hinton2009deep; @hinton2012practical; @goodfellow2014generative; @salakhutdinov2015learning].
These models highlight their capacity to generate data samples using
probabilistic constructs that include DNNs. This new capacity has also
provoked a strong impact within the deep learning community because has
opened the possibility of dealing with unsupervised learning problems,
as opposed to the *classic deep learning methods* which were mainly
focused on supervised learning settings. In any case, this active area
of research is out of the scope of this post and contains many
alternative models which do not fall within the category of models
explored in this work [@goodfellow2014generative].

Probabilistic Programming Languages and Stochastic Computational Graphs
-----------------------------------------------------------------------

One of the main reasons fueling the wide adoption of deep learning has
been the availability of (open-source) software tools containing robust
and well-tested implementations of the main building blocks for defining
and learning DNNs
[@chen2015mxnet; @abadi2016tensorflow; @paszke2017automatic].

Recently, a new wave of software tools is building up on top of these
deep learning frameworks to accommodate modern probabilistic models
containing deep neural networks
[@tran2016edward; @cabanasInferPy; @tran2018simple; @bingham2018pyro].
These software tools usually fall under the umbrella of the so-called
*probabilistic programming languages* (PPLs)
[@gordon2014probabilistic; @ghahramani2015probabilistic], which are
programming languages focused on describing general probabilistic models
and powered by a general inference engine. Although they have been
present in the field of machine learning for many years, this first
generation of PPLs was mainly focused on defining a flexible language to
express probabilistic models which were more general than the
traditional ones usually defined by means of a graphical model
[@koller2009probabilistic]. The advent of deep learning and the
development of probabilistic models containing DNNs has motivated the
development of a new family of PPLs
[@tran2016edward; @cabanasInferPy; @tran2018simple; @bingham2018pyro]
able to define probabilistic models containing DNNs.


The key data structure in these new PPLs are the so-called a *stochastic
computational graphs* (SCGs) [@schulman2015gradient]. *Stochastic
computational graphs* extends standard *computational graphs* with
stochastic nodes. Stochastic nodes are distributed conditionally on
their parents and are represented as circles in the subsequent diagrams.
Stochastic computational graphs allow then to define complex functions
involving expectations over random variables.
Figure [\[fig:StochasticCG\]](#fig:StochasticCG) shows several examples of some simple
stochastic computational graphs depending on a parameter vector
$\lambda$. Modern PPLs offer a wide and diverse range of probability
distributions to define stochastic computational graphs
[@dillon2017tensorflow]. And these probability distributions are defined
over tensors objects.

<a name="fig:StochasticCG" style="float: right;">[]</a>
![](../images/figStochasticCG.png)
*Examples of two computational graphs encoding two different expectations*


We should note that SCGs are not directly implemented within the PPLs,
because computing the exact expected value of complex functions is
usually not feasible. However, they are indirectly coded on top of the
existing computational graph's framework. In this way, each stochastic
node, $\bmz$, is associated with a tensor, $\bmz^\star$, which
represents a (set of) sample(s) from the distribution associated to
$\bmz$. $\bmz^\star$ is the tensor which is fed to the underlying
computational graph. So, SCGs are *simulated* using standard
computational graphs and samples from the defined probability/densities
distributions. Figure
[\[fig:EvaluatingStochasticCG\]](#fig:EvaluatingStochasticCG) illustrates different
possibilities about how SCG can be simulated using standard CGs.

<a name="fig:EvaluatingStochasticCG" style="float: right;">[]</a>
![](../images/EvaluatingStochasticCG.png)
*(**Left**) A stochastic computational graph encoding the function $h=E_{z}[z + 5]$, where $z\sim N(\lambda,1)$. (**Center**) Computational graph processing $k$ samples from $z$ and producing $h^\star$ containing $K$ samples from $N(\lambda+5,1)$. (**Right**) Computational graph processing $k$ samples from $z$ and producing $\hat{h}$, an estimate of  $E_z[z + 5]$. Note that CGs efficiently operate with tensors, with current toolbox like Tensorflow exploiting high-performance computing hardware such as GPUs or TPUs. So, its much more efficient to run the CG once over a bunch of samples, than running multiple times the CG over a single sample.*


Stochastic computational graphs allow defining quite general
probabilistic models, potentially including complex deterministic
relationships as DNNs. All the concepts review in this post applies
to any probabilistic model which can be defined by means of a stochastic
computational graph. Even though, the following equation provides a
general characterization covering most of the models discussed in this
post in terms of a deep LVM. We introduce this
characterization because it will be easier for the reader to trace back
a connection with the models and concepts commented in previous posts.

<a name="eq:dnnexponentialform" style="float: right;">[]</a>
$$\begin{aligned}
\label{eq:dnnexponentialform}
\ln p(\bmbeta) &=& \ln h(\bmbeta) + \bmalpha^T t(\bmbeta) - a_g(\bmalpha)\nonumber\\
\ln p(\bmz_i|\bmbeta) &=& \ln h(\bmz_i)  + \eta_z(\bmbeta)^T t(\bmz_i) - a_z(\eta_z(\bmbeta))\nonumber\\
\bmh_0 &=& a_0(\bmz_i^T\bmbeta_0)\nonumber\\
&\ldots &\nonumber\\
\bmh_{l} &=& a_l(\bmh_{l-1}^T\bmbeta_{l-1})\nonumber\\
&\ldots &\nonumber\\
\bmh_L &=& a_L(\bmh_{L}^T\bmbeta_L)\nonumber\\
\ln p(\bmx_i|\bmz_i,\bmbeta) &=& \ln h(\bmx_i)  + \eta_x(\bmh_L)^T t(\bmx_i) - a_x(\eta_x(\bmh_L)).\end{aligned}$$

As can be seen, the main difference with respect to standard LVMs is the conditional distribution over the
observations $\bmx_i$ given the local hidden variables $\bmz_i$ and the
global parameters $\bmbeta$. Now, this conditional relationship is
governed by the DNN parameterized by $\bmbeta$. The DNN affects how the
hidden variable $\bmz$ defines the parameters of the conditional
distribution $p(\bmx\given\bmz)$. Note that the DNN does not directly
relate $\bmz$ with $\bmx$. A graphical description of this new model can
be found in Figure
[\[fig:DNNModel\]](#fig:DNNModel).

<a name="fig:DNNModel" style="float: right;">[]</a>
![](../images/DNNModel2.png)
*Core of the probabilistic model examined in this post.*

Deep exponential family models [@ranganath2015deep] would be a
straightforward extension of this model family which is not covered here. It includes different hierarchies of latent variables which
gives rise to a more expressive model family. But all the inference
methods revised in this work would apply to these more complex models
with minor adaptations.


<a name="example:PCA:VI" style="float: right;">[]</a>

---
**Example 4: The Generative part of Variational Auto-Encocer**

As commented above,
Variational Auto-encoders are widely adopted LVM containing DNNs
[@kingma2013auto]. Algorithm
[\[alg:vae\]](#alg:vae)
provides a simplified pseudo-code description of the generative model of
a VAE.

<a name="alg:vae" style="float: right;">[]</a>
![](../images/algo3.png)


This model is quite similar to the PCA model presented in
Example [\[example:PCA\]](#example:PCA). The main difference comes from the conditional
distribution of $\bmx_i$. In the PCA model, the mean of the Normal
distribution of $\bmx_i$ depends linearly on $\bmz_i$ through $\bmbeta$.
In the VAE model, the mean depends on $\bmz_i$ through a DNN
parametrized by $\bmbeta$, this DNN is known as the *decoder network* of
the VAE [@kingma2013auto]. Note that the original formulation of this
model also includes another DNN which connects $\bmz_i$ with the
variance of the Normal distribution of $\bmx_i$. It easy to see that
this model belongs to the model family described by
Equations  [\[eq:dnnexponentialform\]](#eq:dnnexponentialform).


---

## References

<div id="ref-abadi2016tensorflow">
<p>Abadi, Martı́n, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. 2016. “Tensorflow: A System for Large-Scale Machine Learning.” In <em>OSDI</em>, 16:265–83.</p>
</div>
<div id="ref-bingham2018pyro">
<p>Bingham, Eli, Jonathan P Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul Szerlip, Paul Horsfall, and Noah D Goodman. 2018. “Pyro: Deep Universal Probabilistic Programming.” <em>arXiv Preprint arXiv:1810.09538</em>.</p>
</div>
<div id="ref-blei2003latent">
<p>Blei, David M, Andrew Y Ng, and Michael I Jordan. 2003. “Latent Dirichlet Allocation.” <em>Journal of Machine Learning Research</em> 3 (Jan): 993–1022.</p>
</div>
<div id="ref-cabanasInferPy">
<p>Cabañas, Rafael, Antonio Salmerón, and Andrés R. Masegosa. 2019. “InferPy: Probabilistic Modeling with Tensorflow Made Easy.” <em>Knowledge-Based Systems</em>.</p>
</div>
<div id="ref-card2017neural">
<p>Card, Dallas, Chenhao Tan, and Noah A Smith. 2017. “A Neural Framework for Generalized Topic Models.” <em>arXiv Preprint arXiv:1705.09296</em>.</p>
</div>
<div id="ref-chen2015mxnet">
<p>Chen, Tianqi, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. 2015. “Mxnet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems.” <em>arXiv Preprint arXiv:1512.01274</em>.</p>
</div>
<div id="ref-chung2015recurrent">
<p>Chung, Junyoung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron C Courville, and Yoshua Bengio. 2015. “A Recurrent Latent Variable Model for Sequential Data.” In <em>Advances in Neural Information Processing Systems</em>, 2980–8.</p>
</div>
<div id="ref-dillon2017tensorflow">
<p>Dillon, Joshua V, Ian Langmore, Dustin Tran, Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt Hoffman, and Rif A Saurous. 2017. “TensorFlow Distributions.” <em>arXiv Preprint arXiv:1711.10604</em>.</p>
</div>
<div id="ref-doersch2016tutorial">
<p>Doersch, Carl. 2016. “Tutorial on Variational Autoencoders.” <em>arXiv Preprint arXiv:1606.05908</em>.</p>
</div>
<div id="ref-ghahramani2015probabilistic">
<p>Ghahramani, Zoubin. 2015. “Probabilistic Machine Learning and Artificial Intelligence.” <em>Nature</em> 521 (7553): 452.</p>
</div>
<div id="ref-goodfellow2014generative">
<p>Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. “Generative Adversarial Nets.” In <em>Advances in Neural Information Processing Systems</em>, 2672–80.</p>
</div>
<div id="ref-gordon2014probabilistic">
<p>Gordon, Andrew D, Thomas A Henzinger, Aditya V Nori, and Sriram K Rajamani. 2014. “Probabilistic Programming.” In <em>Proceedings of the on Future of Software Engineering</em>, 167–81. ACM.</p>
</div>
<div id="ref-gomez2018automatic">
<p>Gómez-Bombarelli, Rafael, Jennifer N Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamı́n Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D Hirzel, Ryan P Adams, and Alán Aspuru-Guzik. 2018. “Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules.” <em>ACS Central Science</em> 4 (2): 268–76.</p>
</div>
<div id="ref-gregor2015draw">
<p>Gregor, Karol, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, and Daan Wierstra. 2015. “Draw: A Recurrent Neural Network for Image Generation.” <em>arXiv Preprint arXiv:1502.04623</em>.</p>
</div>
<div id="ref-hinton2009deep">
<p>Hinton, Geoffrey E. 2009. “Deep Belief Networks.” <em>Scholarpedia</em> 4 (5): 5947.</p>
</div>
<div id="ref-hinton2012practical">
<p>———. 2012. “A Practical Guide to Training Restricted Boltzmann Machines.” In <em>Neural Networks: Tricks of the Trade</em>, 599–619. Springer.</p>
</div>
<div id="ref-hsu2017learning">
<p>Hsu, Wei-Ning, Yu Zhang, and James Glass. 2017. “Learning Latent Representations for Speech Generation and Transformation.” <em>arXiv Preprint arXiv:1704.04222</em>.</p>
</div>
<div id="ref-jiang2016variational">
<p>Jiang, Zhuxi, Yin Zheng, Huachun Tan, Bangsheng Tang, and Hanning Zhou. 2016. “Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering.” <em>arXiv Preprint arXiv:1611.05148</em>.</p>
</div>
<div id="ref-johnson2016composing">
<p>Johnson, Matthew, David K Duvenaud, Alex Wiltschko, Ryan P Adams, and Sandeep R Datta. 2016. “Composing Graphical Models with Neural Networks for Structured Representations and Fast Inference.” In <em>Advances in Neural Information Processing Systems</em>, 2946–54.</p>
</div>
<div id="ref-jolliffe2011principal">
<p>Jolliffe, Ian. 2011. “Principal Component Analysis.” In <em>International Encyclopedia of Statistical Science</em>, 1094–6. Springer.</p>
</div>
<div id="ref-kingma2013auto">
<p>Kingma, Diederik P, and Max Welling. 2013. “Auto-Encoding Variational Bayes.” <em>arXiv Preprint arXiv:1312.6114</em>.</p>
</div>
<div id="ref-koller2009probabilistic">
<p>Koller, Daphne, and Nir Friedman. 2009. <em>Probabilistic Graphical Models: Principles and Techniques</em>. MIT press.</p>
</div>
<div id="ref-kulkarni2015deep">
<p>Kulkarni, Tejas D, William F Whitney, Pushmeet Kohli, and Josh Tenenbaum. 2015. “Deep Convolutional Inverse Graphics Network.” In <em>Advances in Neural Information Processing Systems</em>, 2539–47.</p>
</div>
<div id="ref-linderman2016recurrent">
<p>Linderman, Scott W, Andrew C Miller, Ryan P Adams, David M Blei, Liam Paninski, and Matthew J Johnson. 2016. “Recurrent Switching Linear Dynamical Systems.” <em>arXiv Preprint arXiv:1610.08466</em>.</p>
</div>
<div id="ref-louizos2017causal">
<p>Louizos, Christos, Uri Shalit, Joris M Mooij, David Sontag, Richard Zemel, and Max Welling. 2017. “Causal Effect Inference with Deep Latent-Variable Models.” In <em>Advances in Neural Information Processing Systems</em>, 6446–56.</p>
</div>
<div id="ref-paszke2017automatic">
<p>Paszke, Adam, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. 2017. “Automatic Differentiation in Pytorch.”</p>
</div>
<div id="ref-pless2009survey">
<p>Pless, Robert, and Richard Souvenir. 2009. “A Survey of Manifold Learning for Images.” <em>IPSJ Transactions on Computer Vision and Applications</em> 1: 83–94.</p>
</div>
<div id="ref-pu2016variational">
<p>Pu, Yunchen, Zhe Gan, Ricardo Henao, Xin Yuan, Chunyuan Li, Andrew Stevens, and Lawrence Carin. 2016. “Variational Autoencoder for Deep Learning of Images, Labels and Captions.” In <em>Advances in Neural Information Processing Systems</em>, 2352–60.</p>
</div>
<div id="ref-ranganath2015deep">
<p>Ranganath, Rajesh, Linpeng Tang, Laurent Charlin, and David Blei. 2015. “Deep Exponential Families.” In <em>Artificial Intelligence and Statistics</em>, 762–71.</p>
</div>
<div id="ref-salakhutdinov2015learning">
<p>Salakhutdinov, Ruslan. 2015. “Learning Deep Generative Models.” <em>Annual Review of Statistics and Its Application</em> 2: 361–85.</p>
</div>
<div id="ref-schulman2015gradient">
<p>Schulman, John, Nicolas Heess, Theophane Weber, and Pieter Abbeel. 2015. “Gradient Estimation Using Stochastic Computation Graphs.” In <em>Advances in Neural Information Processing Systems</em>, 3528–36.</p>
</div>
<div id="ref-semeniuta2017hybrid">
<p>Semeniuta, Stanislau, Aliaksei Severyn, and Erhardt Barth. 2017. “A Hybrid Convolutional Variational Autoencoder for Text Generation.” <em>arXiv Preprint arXiv:1702.02390</em>.</p>
</div>
<div id="ref-sohn2015learning">
<p>Sohn, Kihyuk, Honglak Lee, and Xinchen Yan. 2015. “Learning Structured Output Representation Using Deep Conditional Generative Models.” In <em>Advances in Neural Information Processing Systems</em>, 3483–91.</p>
</div>
<div id="ref-tipping1999probabilistic">
<p>Tipping, Michael E, and Christopher M Bishop. 1999. “Probabilistic Principal Component Analysis.” <em>Journal of the Royal Statistical Society: Series B (Statistical Methodology)</em> 61 (3): 611–22.</p>
</div>
<div id="ref-tran2018simple">
<p>Tran, Dustin, Matthew W Hoffman, Dave Moore, Christopher Suter, Srinivas Vasudevan, and Alexey Radul. 2018. “Simple, Distributed, and Accelerated Probabilistic Programming.” In <em>Advances in Neural Information Processing Systems</em>, 7608–19.</p>
</div>
<div id="ref-tran2016edward">
<p>Tran, Dustin, Alp Kucukelbir, Adji B Dieng, Maja Rudolph, Dawen Liang, and David M Blei. 2016. “Edward: A Library for Probabilistic Modeling, Inference, and Criticism.” <em>arXiv Preprint arXiv:1610.09787</em>.</p>
</div>
<div id="ref-xie2016unsupervised">
<p>Xie, Junyuan, Ross Girshick, and Ali Farhadi. 2016. “Unsupervised Deep Embedding for Clustering Analysis.” In <em>International Conference on Machine Learning</em>, 478–87.</p>
</div>
<div id="ref-zhou2015poisson">
<p>Zhou, Mingyuan, Yulai Cong, and Bo Chen. 2015. “The Poisson Gamma Belief Network.” In <em>Advances in Neural Information Processing Systems</em>, 3043–51.</p>
</div>

---

[^1]: Technically, it refers to high-dimensional data which "lives" in a
    low-dimensional manifold [@pless2009survey].
