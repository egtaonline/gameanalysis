.. _cont_approx:

RBF Gaussian Process Continuous Approximation
=============================================

Our basic setup is a game with a set of roles :math:`r \in \mathcal R = [R]`, a set of strategies :math:`s_r \in \mathcal S_r = [S_r] \ne \varnothing` strategies per role, and a number of players per role :math:`n_r > 0`.
For the rest of this analysis we only consider partial profiles, these are profiles omitting a deviating player (alternately, just omitting one player).
The Gaussian process payoff estimator produces a payoff estimate from a normalized partial profile.
A partial profile is simply an assignment of players to strategies less the deviating player, but for the benefit of notation, we will assume that :math:`n_r` is already less the deviating player:

.. math::
    \mathbf x &= \{x_{rs_r}\}_{r \in \mathcal R, s_r \in \mathcal S_r},\ x_{rs_r} \in \mathbb Z_+,\ \sum_{s_r} x_{rs_r} = n_r

The Gaussian process regressor is defined as

.. math::
    \operatorname{payoff}(\mathbf x) &= \sum_j \alpha_j \exp\left\{ -\frac{1}{2} \sum_{r} \sum_{s_r} \frac{(X_{jrs_r} - x_{rs_r})^2}{\ell_{rs_r}^2} \right\}\\
    &= \left( \prod_{r} (2 \pi)^\frac{S_r}{2} \prod_{s_r} \ell_{rs_r} \right) \sum_j \alpha_j \prod_{r} \mathcal N(\mathbf x_r; \mathbf X_{jr}, \mathbf L_r) \\
    &= C \sum_j \alpha_j \prod_{r} \mathcal N(\mathbf x_r; \mathbf X_{jr}, \mathbf L_r),\ C = \prod_{r} (2 \pi)^\frac{S_r}{2} \prod_{s_r} \ell_{rs_r}

where :math:`\mathbf L_r = \operatorname{diag}(\ell_{rs_r}^2)` are the RBF length scales for each role and strategy.
These can all be identical if the length scale is not specific to a particular strategy, but in general, they should vary by role.
:math:`\boldsymbol \alpha = \mathbf K^{-1} \mathbf y` are the Gaussian process weights for each sample.
:math:`\mathbf X` are the training partial profiles, where :math:`j` indexes over training sample.

Our goal is to estimate the expected deviation payoff from a mixture :math:`\mathbf p`:

.. math::
    \operatorname{devpayoff}(\mathbf p) &= \mathbb E_{\mathbf x_r \sim \mathcal M (n_r, \mathbf p_r)} \left[ \operatorname{payoff}(\mathbf x) \right] \\
    &= \sum_{\mathbf x} C \sum_j \alpha_j \prod_{r} \mathcal N(\mathbf x_r; \mathbf X_{jr}, \mathbf L_r) \mathcal M (\mathbf x_r; n_r, \mathbf p_r) \\
    &\approx \int_{\mathbf x} C \sum_j \alpha_j \prod_{r} \mathcal N(\mathbf x_r; \mathbf X_{jr}, \mathbf L_r) \mathcal N \left(\mathbf x_r; n_r \mathbf p_r, n_r \mathbf M_r\right) \\
    &= \int_{\mathbf x} C \sum_j \alpha_j \prod_r \mathcal N \left(\mathbf X_{jr}; n_r \mathbf p_r, \mathbf L_r + n_r \mathbf M_r \right) \mathcal N (\mathbf x_r; \boldsymbol \mu_r, \boldsymbol \Sigma_r) \\
    &= C \sum_j \alpha_j \prod_r \mathcal N \left(\mathbf X_{jr}; n_r \mathbf p_r, \mathbf L_r + n_r \mathbf M_r \right) \\
    &= C \sum_j \alpha_j \prod_r \left| 2 \pi \left(\mathbf L_r + n_r \mathbf M_r\right) \right| ^{-\frac{1}{2}} \exp \left\{ -\frac{1}{2} (\mathbf X_{jr} - n_r \mathbf p_r)^\top \left(\mathbf L_r + n_r \mathbf M_r\right)^{-1} (\mathbf X_{jr} - n_r \mathbf p_r) \right\} \\
    &= \left( \prod_r \left( \prod_{s_r} \ell_{rs_r} \right) \left| \left(\mathbf L_r + \frac{1}{n_r} \mathbf M_r\right) \right| ^{-\frac{1}{2}} \right) \sum_j \alpha_j \exp \left\{ -\frac{1}{2} \sum_r (\mathbf X_{jr} - n_r \mathbf p_r)^\top \left(\mathbf L_r + n_r \mathbf M_r\right)^{-1} (\mathbf X_{jr} - n_r \mathbf p_r) \right\}

The approximation comes from approximating a sum over the multinomial distribution with an integral over a Gaussian approximation.
The next step is to derive simplifications for the determinant and the inverse of :math:`\mathbf L_r + n_r \mathbf M_r`.
First, we need to define a few helpful variables:

.. math::
    d_{rs_r} &= \ell_{rs_r}^2 + n_r p_{rs_r} \\
    \gamma_r &= 1 - n_r \sum_{s_r} \frac{p_{rs_r}^2}{d_{rs_r}}

Then

.. math::
    \left| \mathbf L_r + n_r \mathbf M_r \right| &= \left| \operatorname{diag}_{s_r}(d_{rs_r}) - n_r \mathbf p_r \mathbf p_r^\top \right| \\
    &= \left( 1 - n_r \sum_{s_r} \frac{p_{rs_r}^2}{d_{rs_r}} \right) \prod_{s_r} d_{rs_r} \\
    &= \gamma_r \prod_{s_r} d_{rs_r} \\
    \left( \mathbf L_r + n_r \mathbf M_r \right)^{-1} &= \left( \operatorname{diag}_{s_r}(d_{rs_r}) - n_r \mathbf p_r \mathbf p_r^\top \right)^{-1} \\
    &= \operatorname{diag}_{s_r} \left(\frac{1}{d_{rs_r}}\right) + \frac{n_r}{ 1 - n_r \sum_{s_r} \frac{p_{rs_r}^2}{d_{rs_r}} } \mathbf q_r \mathbf q_r^\top,\ q_{rs_r} = \frac{p_{rs_r}}{d_{rs_r}} \\
    &= \operatorname{diag}_{s_r} \left(\frac{1}{d_{rs_r}}\right) + \frac{n_r}{\gamma_r} \mathbf q_r \mathbf q_r^\top

Plugging these into the equation for :math:`\operatorname{devpayoff}` yields

.. math::
    \operatorname{devpayoff}(\mathbf p) &= \left( \prod_r \gamma_r^{-\frac{1}{2}} \prod_{s_r} \frac{ \ell_{rs_r} }{ \sqrt{d_{rs_r}} } \right) \sum_j \alpha_j \exp \left\{ -\frac{1}{2} \sum_r \left( (\mathbf X_{jr} - n_r \mathbf p_r)^\top \left(\operatorname{diag}_{s_r} \left(\frac{1}{d_{rs_r}}\right) + \frac{n_r}{ \gamma_r } \mathbf q_r \mathbf q_r^\top \right) (\mathbf X_{jr} - n_r \mathbf p_r) \right) \right\} \\
    &= \left( \prod_r \gamma_r^{-\frac{1}{2}} \prod_{s_r} \frac{ \ell_{rs_r} }{ \sqrt{d_{rs_r}} } \right) \sum_j \alpha_j \exp \left\{ -\frac{1}{2} \sum_r \left( \left( \sum_{s_r} \frac{(X_{jrs_r} - n_r p_{rs_r})^2}{d_{rs_r}} \right) + \frac{n_r}{\gamma_r} \left( \sum_{s_r} \frac{p_{rs_r}}{d_{rs_r}} (X_{jrs_r} - n_r p_{rs_r}) \right)^2 \right) \right\}

The derivative with respect to one element of the mixture :math:`p_{\rho i}` is

.. math::
    \frac{\partial}{\partial p_{\rho i}} \operatorname{devpayoff}(\mathbf p) &= -\frac{1}{2} \left( \prod_r \gamma_r^{-\frac{1}{2}} \prod_{s_r} \frac{ \ell_{rs_r} }{ \sqrt{d_{rs_r}} } \right) \left [ \left( \gamma_{\rho}^{-1} \left( \beta_{\rho i}^2 - 1 \right) + \frac{n_{\rho}}{d_{\rho i}} \right) \sum_j \alpha_j \exp \left\{ \cdot \right\} + n_{\rho} \sum_j \alpha_j \exp \left\{ \cdot \right\} \left( \left( \delta_{j \rho} - 1 \right)^2 - \left( \delta_{j \rho} \beta_{\rho i} - \xi_{j \rho i} \right)^2 \right) \right] \\
    \delta_{j \rho} &= \gamma_{\rho}^{-1} \sum_{s_{\rho}} \frac{p_{\rho s_{\rho}}}{d_{\rho s_{\rho}}} \left( X_{j \rho s_{\rho}} - n_{\rho} p_{\rho s_{\rho}} \right) \\
    \beta_{\rho i} &= 1 - \frac{n_{\rho} p_{\rho i}}{d_{\rho i}} \\
    \xi_{j \rho i} &= 1 + \frac{X_{j \rho i} - n_{\rho} p_{\rho i}}{d_{\rho i}}


Definitions, Notations and Identities
-------------------------------------

.. math::
    [N] &= \{i\}_{i=1}^N \\
    \operatorname{diag}_i(a_i) &= \begin{bmatrix} a_1 & 0 & \dots & 0 \\ 0 & a_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & a_n \end{bmatrix} \\
    \mathcal N (\mathbf x; \boldsymbol \mu, \boldsymbol \Sigma) &= | 2 \pi \boldsymbol \Sigma|^{-\frac{1}{2}} \exp\left\{-\frac{1}{2} (\mathbf x - \boldsymbol \mu)^\top \boldsymbol \Sigma^{-1} (\mathbf x - \boldsymbol \mu) \right\} \\
    \mathcal M (\mathbf x; n, \mathbf p) &= \frac{n!}{\prod_i x_i!}\prod_i p_i^{x_i}\text{ if }\sum_i x_i = n \\
    \mathcal M (\mathbf x; n, \mathbf p) &\approx \mathcal N (\mathbf x; n \mathbf p, n \mathbf M),\ \mathbf M = \operatorname{diag}_i(p_i) - \mathbf p \mathbf p^\top \\
    | \operatorname{diag}_i(a_i) | &= \prod_i a_i \\
    \left| \mathbf A + \mathbf u \mathbf v^\top \right| &= (1 + \mathbf v^\top \mathbf A^{-1} \mathbf u) | \mathbf A | \\
    \left| \operatorname{diag}_i(a_i) + \mathbf u \mathbf v^\top \right| &= \left( 1 + \sum_i \frac{u_i v_i}{a_i} \right) \prod_i a_i \\
    (\mathbf A + \mathbf B \mathbf C \mathbf D)^{-1} &= \mathbf A^{-1} - \mathbf A^{-1} \mathbf B ( \mathbf C^{-1} + \mathbf D \mathbf A^{-1} \mathbf B )^{-1} \mathbf D \mathbf A^{-1} \\
    \left(\operatorname{diag}_i(a_i) + c \mathbf b \mathbf b^\top\right)^{-1} &= \operatorname{diag}_i\left(\frac{1}{a_i}\right) - \left(\frac{1}{c} + \sum_i \frac{b_i^2}{a_i} \right)^{-1} \mathbf b^\prime {\mathbf b^\prime}^\top,\ b_i^\prime = \frac{b_i}{a_i} \\
    \mathcal N ( \mathbf x; \boldsymbol \mu_1, \boldsymbol \Sigma_1 ) \mathcal N (\mathbf x; \boldsymbol \mu_2, \boldsymbol \Sigma_2 ) &= \mathcal N (\boldsymbol \mu_1; \boldsymbol \mu_2, \boldsymbol \Sigma_1 + \boldsymbol \Sigma_2 ) \mathcal N (\mathbf x; \boldsymbol \mu_3, \boldsymbol \Sigma_3 )
