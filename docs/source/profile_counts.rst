.. _profile_counts:

Reduction Profile Counting
==========================

This page contains formulae for computing the number of profiles in different game objects.
Independent of scheduler type, there are a few important quantities for describing the number of profiles of a specific scheduler.
:math:`\mathcal R`, the set of roles in the game.
:math:`n_r > 0`, the reduced number of players in role :math:`r`.
:math:`s_r > 0`, the number of strategies in fully explored set for role :math:`r`.
:math:`d_r \ge 0`, the number of deviating strategies for role :math:`r`.
The most relevant function is the number of combinations with repetition denoted as :math:`\left(\!\!{a \choose b}\!\!\right)`, which is the same as :math:`a + b - 1 \choose b`.
We define this to be 0 if any of the arguments to combinations is below 0.

For normal games, and games with a hierarchical reduction, the closed form is relatively simple.

.. math::
   \prod_{r \in \mathcal R} \left(\!\!{s_r \choose n_r}\!\!\right) + \sum_{r \in \mathcal R} d_r \prod_{\rho \in \mathcal R} \left(\!\!{s_\rho \choose n_\rho - \mathbb{I}_{r = \rho}}\!\!\right)

where :math:`\mathbb{I}_{statement}` is the indicator function, one when :math:`statement` is true, zero when otherwise.
The first term is the number of profiles in the "full" game, and the second term is the number of profiles for the deviations.

For deviation preserving reduction games, the form is a little more complicated.

.. math::
   & \sum_{r \in \mathcal R} s_r \prod_{\rho \in \mathcal R} \left(\!\!{s_\rho \choose n_\rho - \mathbb{I}_{r = \rho}}\!\!\right)
   - \sum_{\mathcal P \in \mathbb P(\mathcal R) \setminus \varnothing} \left[ |\mathcal P| - 1 \right] \prod_{r \in \mathcal P} s_r \prod_{r \in \mathcal R \setminus \mathcal P} \left[ \left(\!\!{s_r \choose n_r}\!\!\right) - s_r \right] \\
   & + \sum_{r \in \mathcal R} d_r \prod_{\rho \in \mathcal R} \left(\!\!{s_\rho \choose n_\rho - \mathbb{I}_{r = \rho}}\!\!\right)
   + \sum_{r \in \mathcal R} d_r \sum_{\rho \in \mathcal R} s_\rho \prod_{o \in \mathcal R} \left(\!\!{s_o \choose n_o - \mathbb{I}_{r = o} - \mathbb{I}_{\rho = o}}\!\!\right) \\
   & - \sum_{r \in \mathcal R} d_r \sum_{\mathcal P \in \mathbb P(\mathcal R) \setminus \varnothing} \left[ |\mathcal P| - 1 - \mathbb{I}_{n_r > 1} \mathbb{I}_{r \in \mathcal P} \right] \prod_{\rho \in \mathcal P} {s_\rho}^{1 - \mathbb{I}_{r = \rho} \mathbb{I}_{n_r = 1}} \prod_{\rho \in \mathcal R \setminus \mathcal P} \left[ \left(\!\!{s_\rho \choose n_\rho - \mathbb{I}_{r = \rho}}\!\!\right) - {s_\rho}^{1 - \mathbb{I}_{r = \rho} \mathbb{I}_{n_r = 1}} \right]

where :math:`\mathbb P` denotes the power set.
The first term is the number of payoffs in the full game, which is a slight overestimate of the DPR profiles.
The second term is the amount of overestimate for the number of full game profiles.
The third term is the number of profiles for the deviators' payoffs.
The fourth term is the number of profiles for the non-deviators' payoffs when there is a deviator.
The last term is the amount of overestimate for the deviating profiles.
Note, that for :math:`|\mathcal R|` items in the second and last term have :math:`|P| = 1`, and so are multiplied by 0 and can be omitted when actually computing the summation.
