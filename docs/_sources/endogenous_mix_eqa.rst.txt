.. _endogenous_mix:

Endogenous Mixture Game Equilibria
==================================

This walks through the process of computing the equilibria for an arbitrary mixture between two games.
For this purpose, we we only care about the mixture ratio :math:`t` between the two games, and the payoff to a single player for deviating to strategy :math:`s` when every other agent mixes according to mixture :math:`\mathbf m`, which we'll denote :math:`a_s (\mathbf m)` and :math:`b_s (\mathbf m)` respectively.
This derivation is for the symmetric case where :math:`s \in \mathcal S, |\mathcal S| = S`, but the extension is role-symmetry is trivial.
The shorthand :math:`\dot f` indicates the derivative of :math:`f` with respect to :math:`t` (:math:`\frac{\partial f}{\partial t}`), and following similar patterns, capital letters are matricies, bold lowercase letters are vectors, and regular lowercase letters are scalars.

For the game where each payoff is :math:`1 - t` fraction of the payoff from game :math:`a` and :math:`t` fraction from game :math:`b`, the deviation payoff is simply :math:`p_s (\mathbf m) = (1 - t) a_s (\mathbf m) + t b_s (\mathbf m)`, which we can differentiate easily:

.. math::
  \dot p_s(\mathbf m) &= b_s(\mathbf m) - a_s(\mathbf m) + (1 - t) \nabla_{\mathbf m}^\top a_s(\mathbf m) \dot{\mathbf m} + t \nabla_{\mathbf m}^\top b_s(\mathbf m) \dot{\mathbf m} \\
  &= g_s(\mathbf m) + \mathbf f_s^\top(\mathbf m, t) \dot{\mathbf m} \\
  g_s(\mathbf m) &= b_s(\mathbf m) - a_s(\mathbf m) \\
  \mathbf f_s(\mathbf m) &= (1 - t) \nabla_{\mathbf m} a_s(\mathbf m) + t \nabla_{\mathbf m} b_s(\mathbf m)

If :math:`\mathbf m` is an equilibrium mixture of game :math:`p`, then for every pair of strategies :math:`r,s` with support (:math:`m_i > 0, i \in {r,s}`) that :math:`p_r(\mathbf m) = p_s(\mathbf m)`, which also implies :math:`\dot p_r(\mathbf m) = \dot p_s(\mathbf m)`.
:math:`S - 1` independent pairs of strategies can chosen to generate :math:`S - 1` independent equalities of the form:

.. math::
  g_r(\mathbf m) - g_s(\mathbf m) &= \left[ \mathbf f_s^\top(\mathbf m, t) - \mathbf f_r^\top(\mathbf m, t) \right] \dot{\mathbf m}

A simple choice of pairs is :math:`\{(s, i) | i \in \mathcal S \setminus \{ s \}\}`.
The final equation keeps the equilibrium mixture a mixture, :math:`\mathbf 1^\top \dot{\mathbf m} = 0`.
If we take the strategies as indices from :math:`1` to :math:`S` and use the pairs suggested, this can be represented as the matrix equation:

.. math::
  \begin{bmatrix} g_1(\mathbf m) - g_2(\mathbf m) \\ \vdots \\ g_1(\mathbf m) - g_S(\mathbf m) \\ 0 \end{bmatrix} &= \begin{bmatrix} \mathbf f_1^\top(\mathbf m, t) - \mathbf f_2^\top(\mathbf m, t) \\ \vdots \\ \mathbf f_1^\top(\mathbf m, t) - \mathbf f_S^\top(\mathbf m, t) \\ \mathbf 1^\top \end{bmatrix} \dot{\mathbf m} \\
  \mathbf g(\mathbf m) &= \mathbf F(\mathbf m, t) \dot{\mathbf m} \\
  \dot{\mathbf m} &= \mathbf F^{-1}(\mathbf m, t) \mathbf g(\mathbf m)

This final equation represents the derivative of the components of an equilibrium mixture with support in a :math:`t` game.
Given an equilibrium of a :math:`t`, equilibria of nearby games can be found using numeric ODE solving techniques until either a beneficial deviation exists to a strategy outside of support, or support for a current strategy drops to zero.
In the later case, the support can just be dropped, and the technique can progress, in the former, this equilibrium effectively disappears, and a new equilibrium must be found for :math:`t \pm \epsilon`.

A limitation of this method is that it stops as soon as beneficial deviation exists outside of support.
There are papers the project the equilibrium into a space where the equilibria are piecewise continuous, and as a result, may skirt the need to find new equilibria when a beneficial deviation exists outside of support, however, these methods aren't readily applicable to our circumstances for two reasons.

a) They parameterize games by subtracting off the expected deviation payoff played against the uniform mixture, which we generally can't sample because it would imply having complete game data instead of sparse game data.
b) The projection of the games does not merge arbitrary games :math:`a` and :math:`b`, but instead ones that differ only in the expected payoff to deviating from the uniform mixture, meaning that some aspect of the projection would need to be tweaked in order to work for arbitrary games.
