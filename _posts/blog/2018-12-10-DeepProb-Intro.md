---
layout: post-light-feature
title: Deep Probabilistic Modeling (I). A review of traditional inference methods
description: Why traditional probabilistic modeling focused on capture linear relationships among random variables. 
category: articles
tags: [probabilistic models, probability models, exponential families, variational inference, autoencoder, neural networks, graphical models, inference, deep learning, machine learning]
image:
  feature: bayestheorem.jpg
  thumb: bayestheorem.jpg
published: true
---
The seminal works of Judea Pearl [@Pearl88] and Stephen Lauritzen
[@lauritzen1992propagation] about probabilistic graphical models (PGMs)
placed probabilistic modeling as an indispensable tool for dealing with
many problems involving any form of uncertainty within many different
fields such as artificial intelligence [@russell2016artificial],
statistics [@HastieTibshiraniFriedman01], machine learning
[@bishop2006pattern; @murphy2012machine], etc. PGMs has been present for
the last 30 years becoming a well established and highly influential
body of research.

At the same time, the inference problem
[@Pearl88; @lauritzen1992propagation; @JensenNielsen07; @koller2009probabilistic],
as the problem of computing the posterior probability over hidden
quantities given the known evidence, has been the corner-stone (and the
bottleneck) of the feasibility and applicability of probabilistic
modeling.

In the beginning, the first proposed inference algorithms
[@Pearl88; @lauritzen1992propagation] were able to compute this
posterior in an exact way by exploiting the conditional independence
relationships encoded by the graphical structure of the model. Even
though, model's probability distributions were strongly restricted
mainly to multinomial and (conditional linear) Gaussian distributions
[@JensenNielsen07; @koller2009probabilistic]. But researchers quickly
realized these exact inferences schemes were not powerful enough to deal
with complex stochastic dependency structures that arise in many
relevant problems. Mainly due to the high computational costs associated
to the inference algorithms [@koller2009probabilistic]. In consequence,
approximated inference methods started to be the main focus of research.

Monte-Carlo methods were one of the first approximate methods employed
to make inference over complex PGMs
[@gilks1995markov; @plummer2003jags]. They are extremely powerful and
able to approximate complex posterior distributions. However, they have
serious issues like problems of convergence of the underlying Markov
chain, poor mixing, etc. when having to approximate highly dimensional
posteriors [@gilks1995markov]. And computing these highly-dimensional
posteriors started to be relevant in many domains, specially when
researchers seek to apply a Bayesian approach to learn the parameters of
their PGMs from data
[@bishop2006pattern; @murphy2012machine; @blei2014build]. In this case,
the model parameters are treated as unobserved random variables, and the
learning problem reduces to compute the posterior probability over them.
For models with a large number of parameters results in highly
dimensional posteriors where the application of Monte-Carlo methods
became infeasible. And these issues gave rises to the development of
alternative approximate inference schemes.

Belief propagation (BP) [@Pearl88; @murphy1999loopy], and the close
scheme called Expectation propagation (EP) [@minka2001expectation], has
been successfully used in many applications of PGMs helping to overcome
many of the limitations of Monte-Carlo methods. They are approximate
deterministic inference techniques which can be implemented using a
message-passing scheme which exploits the graph structure of the PGM
and, hence, the underlying conditional independence relationships among
variables. In terms of distributional assumptions, BP was mainly
restricted to multinomial and Gaussian distributions, while EP allows
for a more general family of distributions, although restricted by the
need to define a non-trivial quotient operation between the involved
densities. As already commented, these techniques (a many variations
also published later) are deterministic and overcame some of the
difficulties of Monte-Carlo methods. However they presented two main
issues: they did not guarantee the convergence to an approximate and
meaningful solution; and did not scale to the kind of models that appear
in the context of Bayesian learning (i.e. plateau like models)
[@murphy2012machine; @blei2014build]. Again, these issues motivated
researchers to look into in alternative approximate inference schemes.

Variational methods [@wainwright2008graphical] were firstly explored in
the late 90s [@jordan1999introduction], inspired by their successful
application in inference problems encountered in statistical physics.
They are deterministic approximate inference techniques like BP and EP
methods. Their main innovation comes from casting the inference problem
as the problem of maximizing a well defined loss function (i.e. the ELBO
function) acting as an inference proxy. In general, variational methods
guarantee convergence to a local minimum of this ELBO function, and,
then, to a meaningful solution. By transforming the inference problem in
a continuous optimization problem, variational methods could take
advantage of recent advances in continuous optimization theory. That was
the case of the widely adopted stochastic gradient descent algorithm
[@bottou2010large], which was successfully used by the machine learning
community to scale their learning algorithms to big data sets. This same
learning algorithm was adapted to the variational inference problem
[@JMLR:v14:hoffman13a], giving the opportunity to apply probabilistic
modeling approaches to problems involving massive data sets. But, in
terms of distributional assumptions, VI methods were tightly restricted
to the conjugate exponential family [@barndorff2014information], where
ELBO's gradients can be computed in closed-form [@WinnBishop05]. Ad-hoc
approaches were developed over the years for non-conjugate models.

Then, since the start of the field at end of the eighties, probabilistic
models has been mainly focused on exploiting conditional independencies
among random variables and modelling the dependencies using
distributions belonging to the conjugate exponential family. But
exponential family distributions are only able to model linear
relationships between the random variables [@WinnBishop05]. The recent
success of deep learning [@goodfellow2016deep] has been partly due to
the capacity of deep neural networks to model highly non-linear
relationships among highly-dimensional objects as happens between the
pixels of an image or the words of a document, to name just the most
known examples.

Recent advances in variational inference
[@kingma2013auto; @ranganath2014black] gave the opportunity to introduce
in probabilistic models deep neural networks to capture non-linear
relationships among random variables, giving rise to a whole new family
of probabilistic models, which are mainly known as *deep generative
models*
[@hinton2009deep; @hinton2012practical; @goodfellow2014generative; @salakhutdinov2015learning],
which is a very active field of research. This new family of
probabilistic models are able to model in a probabilistic manner objects
like images, text, audio, video, etc. in a much powerful manner than
before, by bringing to the probabilistic modeling field many of the
recent advances produced by the deep learning community. The release of
modern probabilistic programming languages
[@tran2016edward; @cabanasInferPy; @tran2018simple; @bingham2018pyro]
relying on well established deep learning engines
[@hinton2009deep; @hinton2012practical; @goodfellow2014generative; @salakhutdinov2015learning]
are also greatly expanding the adoption of these powerful probabilistic
modeling techniques.

References
-----------
<div id="ref-barndorff2014information">
<p>Barndorff-Nielsen, Ole. 2014. <em>Information and Exponential Families: In Statistical Theory</em>. John Wiley &amp; Sons.</p>
</div>
<div id="ref-bingham2018pyro">
<p>Bingham, Eli, Jonathan P Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul Szerlip, Paul Horsfall, and Noah D Goodman. 2018. “Pyro: Deep Universal Probabilistic Programming.” <em>arXiv Preprint arXiv:1810.09538</em>.</p>
</div>
<div id="ref-bishop2006pattern">
<p>Bishop, Christopher M. 2006. <em>Pattern Recognition and Machine Learning</em>. springer.</p>
</div>
<div id="ref-blei2014build">
<p>Blei, David M. 2014. “Build, Compute, Critique, Repeat: Data Analysis with Latent Variable Models.” <em>Annual Review of Statistics and Its Application</em> 1: 203–32.</p>
</div>
<div id="ref-bottou2010large">
<p>Bottou, Léon. 2010. “Large-Scale Machine Learning with Stochastic Gradient Descent.” In <em>Proceedings of Compstat’2010</em>, 177–86. Springer.</p>
</div>
<div id="ref-cabanasInferPy">
<p>Cabañas, Rafael, Antonio Salmerón, and Andrés R. Masegosa. 2019. “InferPy: Probabilistic Modeling with Tensorflow Made Easy.” <em>Knowledge-Based Systems</em>.</p>
</div>
<div id="ref-gilks1995markov">
<p>Gilks, Walter R, Sylvia Richardson, and David Spiegelhalter. 1995. <em>Markov Chain Monte Carlo in Practice</em>. Chapman; Hall/CRC.</p>
</div>
<div id="ref-goodfellow2016deep">
<p>Goodfellow, Ian, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. 2016. <em>Deep Learning</em>. Vol. 1. MIT press Cambridge.</p>
</div>
<div id="ref-goodfellow2014generative">
<p>Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. “Generative Adversarial Nets.” In <em>Advances in Neural Information Processing Systems</em>, 2672–80.</p>
</div>
<div id="ref-HastieTibshiraniFriedman01">
<p>Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2001. <em>The Elements of Statistical Learning</em>. New York, NY, USA: Springer New York Inc.</p>
</div>
<div id="ref-hinton2009deep">
<p>Hinton, Geoffrey E. 2009. “Deep Belief Networks.” <em>Scholarpedia</em> 4 (5): 5947.</p>
</div>
<div id="ref-hinton2012practical">
<p>———. 2012. “A Practical Guide to Training Restricted Boltzmann Machines.” In <em>Neural Networks: Tricks of the Trade</em>, 599–619. Springer.</p>
</div>
<div id="ref-JMLR:v14:hoffman13a">
<p>Hoffman, Matthew D., David M. Blei, Chong Wang, and John Paisley. 2013. “Stochastic Variational Inference.” <em>Journal of Machine Learning Research</em> 14: 1303–47.</p>
</div>
<div id="ref-JensenNielsen07">
<p>Jensen, Finn V., and Thomas D. Nielsen. 2007. <em>Bayesian Networks and Decision Graphs</em>. Berlin, Germany: Springer-Verlag.</p>
</div>
<div id="ref-jordan1999introduction">
<p>Jordan, Michael I, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. 1999. “An Introduction to Variational Methods for Graphical Models.” <em>Machine Learning</em> 37 (2): 183–233.</p>
</div>
<div id="ref-kingma2013auto">
<p>Kingma, Diederik P, and Max Welling. 2013. “Auto-Encoding Variational Bayes.” <em>arXiv Preprint arXiv:1312.6114</em>.</p>
</div>
<div id="ref-koller2009probabilistic">
<p>Koller, Daphne, and Nir Friedman. 2009. <em>Probabilistic Graphical Models: Principles and Techniques</em>. MIT press.</p>
</div>
<div id="ref-lauritzen1992propagation">
<p>Lauritzen, Steffen L. 1992. “Propagation of Probabilities, Means, and Variances in Mixed Graphical Association Models.” <em>Journal of the American Statistical Association</em> 87 (420): 1098–1108.</p>
</div>
<div id="ref-minka2001expectation">
<p>Minka, Thomas P. 2001. “Expectation Propagation for Approximate Bayesian Inference.” In <em>Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence</em>, 362–69. Morgan Kaufmann Publishers Inc.</p>
</div>
<div id="ref-murphy2012machine">
<p>Murphy, Kevin P. 2012. <em>Machine Learning: A Probabilistic Perspective</em>. MIT press.</p>
</div>
<div id="ref-murphy1999loopy">
<p>Murphy, Kevin P, Yair Weiss, and Michael I Jordan. 1999. “Loopy Belief Propagation for Approximate Inference: An Empirical Study.” In <em>Proceedings of the Fifteenth Conference on Uncertainty in Artificial Intelligence</em>, 467–75. Morgan Kaufmann Publishers Inc.</p>
</div>
<div id="ref-Pearl88">
<p>Pearl, Judea. 1988. <em>Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference</em>. San Mateo, CA.: Morgan Kaufmann Publishers.</p>
</div>
<div id="ref-plummer2003jags">
<p>Plummer, Martyn, and others. 2003. “JAGS: A Program for Analysis of Bayesian Graphical Models Using Gibbs Sampling.” In <em>Proceedings of the 3rd International Workshop on Distributed Statistical Computing</em>. Vol. 124. 125.10. Vienna, Austria.</p>
</div>
<div id="ref-ranganath2014black">
<p>Ranganath, Rajesh, Sean Gerrish, and David Blei. 2014. “Black Box Variational Inference.” In <em>Artificial Intelligence and Statistics</em>, 814–22.</p>
</div>
<div id="ref-russell2016artificial">
<p>Russell, Stuart J, and Peter Norvig. 2016. <em>Artificial Intelligence: A Modern Approach</em>. Malaysia; Pearson Education Limited,</p>
</div>
<div id="ref-salakhutdinov2015learning">
<p>Salakhutdinov, Ruslan. 2015. “Learning Deep Generative Models.” <em>Annual Review of Statistics and Its Application</em> 2: 361–85.</p>
</div>
<div id="ref-tran2018simple">
<p>Tran, Dustin, Matthew W Hoffman, Dave Moore, Christopher Suter, Srinivas Vasudevan, and Alexey Radul. 2018. “Simple, Distributed, and Accelerated Probabilistic Programming.” In <em>Advances in Neural Information Processing Systems</em>, 7608–19.</p>
</div>
<div id="ref-tran2016edward">
<p>Tran, Dustin, Alp Kucukelbir, Adji B Dieng, Maja Rudolph, Dawen Liang, and David M Blei. 2016. “Edward: A Library for Probabilistic Modeling, Inference, and Criticism.” <em>arXiv Preprint arXiv:1610.09787</em>.</p>
</div>
<div id="ref-wainwright2008graphical">
<p>Wainwright, Martin J, Michael I Jordan, and others. 2008. “Graphical Models, Exponential Families, and Variational Inference.” <em>Foundations and Trends in Machine Learning</em> 1 (1–2): 1–305.</p>
</div>
<div id="ref-WinnBishop05">
<p>Winn, John M., and Christopher M. Bishop. 2005. “Variational Message Passing.” <em>Journal of Machine Learning Research</em> 6: 661–94.</p>
</div>

