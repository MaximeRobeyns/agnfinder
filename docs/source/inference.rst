.. _inference:
.. sectionauthor:: Maxime Robeyns <maximerobeyns@gmail.com>

Inferring Galaxy Parameters
###########################

The original motivation behind the *AGNFinder* project was to speed up Bayesin
SED fitting.
MCMC approaches need to evaluate the likelihood term :math:`p(x \vert \theta)`
many times which is a slow process, dominated by the evaluation of the forward
model :math:`f : \Theta \to \mathcal{X}`. This is a mapping from physical galaxy
parameters :math:`\theta \in \Theta` (such as mass, star formation, E(B-V)
as well as AGN signatures such as AGN disc and torus, disk inclination and so
forth) to (mock) photometric observations :math:`x \in \mathcal{X}`.

To speed up the evaluation of the likelihood, we can *emulate* the forward model
with some function approximator (for instance a GP or a neural network). This is
the approach taken in Alsing *et al.* [SPEC2020]_, with good results, and was
incidentally the original goal of this project.

In this fork we take a slightly different approach to recovering physical
parameters from photometric observations :math:`p(\theta \vert x)`. We first
direct our attention from emulating (and speeding up) the forward model, to the
main objective of recovering the physical parameters. We also eschew the MCMC
methods used in this step in favour of a variational Bayesian attack; namely a
conditional VAE---a deep conditional generative model with latent variables.

We motivate the use of this model by acknowledging that the low-dimensional
photometric observations (8 for the Euclid survey) are potentially weakly
predictive of the free galaxy parameters :math:`\theta`; particularly if
:math:`\theta` is relatively high dimensional. We are therefore trying to learn
a 'few-to-many' mapping where the conditional distribution :math:`p(\theta \vert
x)` is complicated and multi-modal.

If we were to use a discriminative model (such as a conventional feedforward
neural network, directly learning the mapping :math:`f: \mathcal{X} \to \Theta`)
then we would merely be making use of *correlations* in the dataset of simulated
:math:`(\theta, x)` pairs; :math:`\mathcal{D} = \{(\theta_{i},
x_{i})\}_{i=1}^{n}` to make predictions.

Attempting to model the generative process by using a CVAE may allow us to
uncover causal relations [IVAE2019]_ in an unsupervised manner, using only the
simulated dataset :math:`\mathcal{D}`. This may make this approach more robust
to extrapolation, and use in different surveys.

..
    A generative model on the other hand learns the distribution of the
    predictor and response jointly; that is:

    .. math::

        p(\mathcal{D}) \stackrel{iid.}{=} \prod_{i=1}^{n}p(x_{i}, \theta_{i}).

    Drawing samples from this (learned) data distribution :math:`p(\mathcal{D})`
    would yield plausible-looking galaxy photometry along with their physical
    parameters.

    To recover a discriminative model :math:`p(\theta \vert x)`, we apply Bayes
    rule, and optimise the evidence lower bound (ELBO) as a substitute for
    evaluating the generally intractable marginal likelihood or *evidence* term in
    the denominator.

(Conditional) Variational Autoencoders
--------------------------------------

.. note:: To avoid a clash of notation, we will henceforth denote the physical
   galaxy parameters as :math:`y` (previously :math:`\theta`). This matches
   the machine learning nomenclature of denoting outputs to be predicted as
   :math:`y`.

Latent Variable Models
~~~~~~~~~~~~~~~~~~~~~~

A variational autoencoder (VAE) is an example of a *latent variable model*
(LVM). Latent variables, often denoted :math:`z`, are unobserved variables which
ideally represent some disentangled, semantically meaningful, and statistically
independent causal factors for variation in the data :math:`x`.

A latent variable model is one of the form

.. math::
   \begin{align*}
   p_{\theta_{z}}(z) &= f_{z}(z; \theta_{z}) \\
   p_{\theta_{y}}(y \vert z) &= f_{y}(y; z, \theta_{y}),
   \end{align*}

where :math:`f_{z}` and :math:`f_{y}` are valid distribution functions, and
:math:`\theta = \{\theta_{z}, \theta_{y}\}` parametrises the generative process.
To sample from :math:`p_{\theta_{y}}(y \vert z)` (and generate plausible
galaxy parameters), we first sample from the 'prior' over the latent variables
:math:`z_{i} \sim p_{\theta_{z}}(z)`, and condition on this :math:`y_{i} \sim
p_{\theta_{y}}(y \vert z_{i})`. In practice, :math:`f_{z}` and :math:`f_{y}` are
neural networks.

In the *conditional* VAE, we are interested in learning a model of galaxy
parameters conditioned on photometric observations. Using :math:`p^{*}(\cdot)`
to denote the true underlying data distribution, we seek

.. math::

    p_{\theta}(y \vert x) \approx p^{*}(y \vert x) \stackrel{iid.}{=}
    \prod_{(y', x') \in \mathcal{D}} p^{*}(y' \vert x').

In other words, we want to minimise :math:`D\big[p_{\theta}(y \vert x) \Vert
p^{*}(y \vert x)\big]`, where :math:`D` is some suitable distance measure. By
analogy to the above, the conditional latent variable model is of the form

.. math::
   \begin{align*}
   p_{\theta_{z}}(z \vert x) &= f_{z}(z; x, \theta_{z}) \\
   p_{\theta_{y}}(y \vert z, x) &= f_{y}(y; z, x, \theta_{y}).
   \end{align*}

Thus we condition the distribution over the latent variable :math:`z` on the
photometric observations :math:`x`. In turn, we condition the distribution over
the physical galaxy parameters :math:`y` on both the (conditional) latent
samples and the photometric observations.

If our objective is to find some :math:`\theta \in \Theta` such that
:math:`p_{\theta}(y \vert x) \approx p^{*}(y \vert x)`, then we seek to maximise
the (log) marginal likelihood of the :math:`n` training observations under our
model:

.. math::

    \underset{\theta \in \Theta}{\mathrm{argmax}} \sum_{i=1}^{n} \log p_{\theta}(y_{i} \vert x_{i})
    = \underset{\theta \in \Theta}{\mathrm{argmax}} \sum_{i=1}^{n} \log
    \int_{\mathcal{Z}} p_{\theta}(y_{i} \vert z, x_{i}) dz,

Integrating out the latent variable from the LVM :math:`p_{\theta}(y \vert z,
x)` to find the marginal likelihood (or *model evidence*) is often intractable.
We instead optimise a lower-bound on the intractable model evidence (the
*evidence lower bound*, ELBO). Here we introduce an approximate posterior
distribution over the latent variables :math:`q_{\phi}(z \vert y, x) \approx
p_{\theta}(z \vert y, x)`, which is parametrised by :math:`\phi` and should be
convenient to sample from.

We will derive this bound twice since these offer different intuitions.
Beginning with the importance sampling approach, we take the expectation wrt the
approximate distribution :math:`q_{\phi}(z \vert y, x)` on both sides of the
above (first line below), and introduce it on the right as a ratio of itself
(second line) while applying Bayes rule to rearrange :math:`p_{\theta}(y \vert
z, x)` (also second line):

.. math::
   \log p_{\theta}(y \vert x) &=
   \int_{\mathcal{Z}} q_{\phi}(z \vert y, x) \log p_{\theta}(y \vert z, x)dz \\
   &= \int_{\mathcal{Z}} q_{\phi}(z \vert y, x) \left(
   \log \frac{p_{\theta}(y, z \vert x)}{q_{\phi}(z \vert y, x)} +
   \log \frac{q_{\phi}(z \vert y, x)}{p_{\theta}(z \vert y, x)}
   \right) dz \\
   &= \underbrace{\mathbb{E}_{q_{\phi}(z \vert y, x)}\left[
   \log p_{\theta}(y, z \vert x) - \log q_{\phi}(z \vert y, x)
   \right]}_{\text{variational lower-bound, } \mathcal{L}(\theta, \phi; x, y)} +
   D_{\text{KL}}\left[q_{\phi}(z \vert y, x) \Vert p_{\theta}(z \vert y, x)\right].

Since the KL divergence is non-negative, the :math:`\mathcal{L}(\theta, \phi; x,
y)` term indeed lower-bounds the evidence:

.. math::

   \log p_{\theta}(y \vert x) &\ge
   \mathbb{E}_{q_{\phi}(z \vert y, x)} \left[
    \log p_{\theta}(y \vert z, x) + \log p_{\theta}(z \vert x) - \log q_{\phi}(z \vert y, x)
    \right] \\
   &= \mathbb{E}_{q_{\phi}(z \vert y, x)}\left[
    \log p_{\theta}(y \vert z, x)
    \right] + \int_{\mathcal{Z}} q_{\phi}(z \vert y, x) \log \frac{p_{\theta}(z \vert
     x)}{q_{\phi}(z \vert y, x)} dz \\
     &= \mathbb{E}_{q_{\phi}(z \vert y, x)}\left[\log p_{\theta}(y \vert z, x)\right]
     - D_{\text{KL}}\left[q_{\phi}(z \vert y, x) \Vert p_{\theta}(z \vert x)\right].

This last line above is the canonical form in which the ELBO is usually
presented.

.. sidebar:: Jensen's inequality

    .. image:: ./_static/jensens-inequality.svg

    Put loosely, Jensen's inequality states that :math:`\varphi(\mathbb{E}[x])
    \le \mathbb{E}[\varphi(x)]`, for :math:`\varphi(\cdot)` a convex function.

For another perspective, we may derive the lower bound using Jensen's
inequality.

In the first line below, we explicitly write the marginalisation
over the latents :math:`z`, and we also introduce the encoder or *recognition
model* :math:`q_{\phi}(z \vert y, x)`. On the second line, we use Jensen's
inequality to push the logarithm inside the expectation, and introduce the lower
bound.

.. math::

   \log p_{\theta}(y \vert x) &= \log \int_{\mathcal{Z}} p_{\theta}(y, z \vert x) \frac{q_{\phi}(z \vert y, x)}{q_{\phi}(z \vert y, x)} dz \\
   &\ge \int_{\mathcal{Z}}q_{\phi}(z \vert y, x)\big(\log p_{\theta}(y, z \vert x)
   - \log q_{\phi}(z \vert y, x)\big) dz \\
     &= \mathbb{E}_{q_{\phi}(z \vert y, x)}\left[\log p_{\theta}(y, z \vert x) - \log q_{\phi}(z \vert y, x)\right] \\
     &\doteq \mathcal{L}(\theta, \phi; x, y).

We can now perform the same rearrangements as above on
:math:`\mathcal{L}(\theta, \phi; x, y)` to reach the canonical expression for
the ELBO.

In order to optimise this objective over both :math:`\theta` and :math:`\phi`
using SGD, we apply the reparametrisation trick and a Monte Carlo approximation
to the expectations, giving:

.. math::

   \mathcal{L}_{\text{CVAE}}(\theta, \phi; x, y) = \frac{1}{K}\sum_{i=1}^{K}
   \log p_{\theta}(y \vert z^{(i)}, x) - D_{\text{KL}}\left[q_{\phi}(z \vert y,
   x) \Vert p_{\theta}(z \vert x)\right],

where :math:`z^{(i)} = g_{\phi}(y, x, \epsilon^{(i)})`, :math:`\epsilon^{(i)}
\sim \mathcal{N}(\mathbf{0}, \mathbf{I})` and :math:`K` is the number of samples
in the empirical expectation.


References
----------

.. [SPEC2020] Alsing Justin, Hiranya Peiris, Joel Leja, ChangHoon Hahn, Rita
   Tojeiro, Daniel Mortlock, Boris Leistedt, Benjamin D. Johnson, and Charlie
   Conroy. ‘SPECULATOR: Emulating Stellar Population Synthesis for Fast and
   Accurate Galaxy Spectra and Photometry’. The Astrophysical Journal Supplement
   Series 249, no. 1 (26 June 2020): 5.
   https://doi.org/10.3847/1538-4365/ab917f.



.. [IVAE2019] Kingma, Diederik P., and Max Welling. ‘An Introduction to
   Variational Autoencoders’. Foundations and Trends® in Machine Learning 12,
   no. 4 (2019): 307–92. https://doi.org/10.1561/2200000056.

