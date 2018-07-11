.. _profile_nash:

Nash Equilibrium Methods Comparison
===================================

For each method available for Nash equilibrium finding, this lists various
information about the performance across different game types and starting
locations. "Fraction of Eqa" is the mean fraction of all equilibria found via
that method or starting location. "Weigted Fraction (of Eqa)" is the same,
except each equilibrium is down weighted by the number of methods that found
it, thus a larger weighted fraction indicates that this method found more
unique equilibria. "Time" is the average time in seconds it took to run this
method for every starting location. "Normalized Time" sets the minimum time for
each game type and sets it to one, thus somewhat mitigating the fact that
certain games may be more difficult than others. It also provides an easy
comparison metric to for baseline timing.

Comparisons Between Methods
----------------------------------

=============================  =================  ===================  ============  =================
Method                           Fraction of Eqa    Weighted Fraction    Time (sec)    Normalized Time
=============================  =================  ===================  ============  =================
Regret Minimization                     0.568731            0.173796       0.321582            9.26718
Multiplicative Weights Bandit           0.528874            0.153923      29.4978           4398.16
Multiplicative Weights Dist             0.521398            0.0959819     17.8935            768.361
Multiplicative Weights Stoch            0.508405            0.0873798     62.5403           7982.75
Fictitious Play                         0.472796            0.0594228      7.73066           576.771
Fictitious Play Long                    0.466721            0.0552945    179.604           22325.4
Replicator Dynamics                     0.438658            0.0567802      1.56668            64.2687
Scarf 30                                0.432357            0.071016      16.7026            309.018
Scarf 5                                 0.431933            0.0708725      5.81228           144.357
Scarf 1                                 0.43032             0.070612       2.01623            61.1935
Regret Matching                         0.370006            0.104922      57.9059           5180.56
=============================  =================  ===================  ============  =================

Scarf 30
--------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                   0.994774             0.994774
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Scarf 1                                 1            5.04985
Scarf 5                                 1            2.14065
Replicator Dynamics                     0.892422     4.80821
Multiplicative Weights Bandit           0.889781     0.0702606
Multiplicative Weights Stoch            0.885552     0.0387107
Regret Minimization                     0.879359    33.3454
Fictitious Play                         0.878377     0.535772
Fictitious Play Long                    0.877982     0.0138415
Multiplicative Weights Dist             0.875912     0.402178
Regret Matching                         0.498422     0.0596495
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.5
Weighted Fraction of Eqa  0.0555556
Time (sec)                0.000148058
Normalized Time           1
========================  ===========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.331322
Weighted Fraction of Eqa    0.074747
Time (sec)                 10.8339
Normalized Time           661.379
========================  ==========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.5375
Weighted Fraction of Eqa  0.0772222
Time (sec)                0.00523914
Normalized Time           1
========================  ==========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.312472
Weighted Fraction of Eqa    0.0581924
Time (sec)                  1.61499
Normalized Time           120.84
========================  ===========

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.8
Weighted Fraction of Eqa  0.0954167
Time (sec)                0.00686474
Normalized Time           1.24652
========================  ==========

Sineagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa            0.5
Weighted Fraction of Eqa   0.05
Time (sec)                 1.53651
Normalized Time           67.9854
========================  ========

Normagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.469426
Weighted Fraction of Eqa    0.0632558
Time (sec)                 40.3947
Normalized Time           634.09
========================  ===========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.0178571
Weighted Fraction of Eqa  0.00595238
Time (sec)                0.0364892
Normalized Time           1.00035
========================  ==========

Local effect
""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.576389
Weighted Fraction of Eqa  0.106637
Time (sec)                0.189833
Normalized Time           6.63798
========================  ========

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.453237
Weighted Fraction of Eqa   0.0686254
Time (sec)                 1.63167
Normalized Time           84.8835
========================  ==========

Covariant
"""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.155377
Weighted Fraction of Eqa     0.0352583
Time (sec)                  87.0647
Normalized Time           2682.6
========================  ============

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.333333
Weighted Fraction of Eqa  0.0333333
Time (sec)                0.000152111
Normalized Time           1
========================  ===========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.525
Weighted Fraction of Eqa  0.148674
Time (sec)                0.00902817
Normalized Time           1.5135
========================  ==========

Normagg large
"""""""""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa              0
Weighted Fraction of Eqa     0
Time (sec)                1800
Normalized Time           1560.87
========================  =======

Sineagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.239765
Weighted Fraction of Eqa  0.0375492
Time (sec)                0.687516
Normalized Time           4.65944
========================  =========

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.0588235
Weighted Fraction of Eqa   0.0196078
Time (sec)                 0.484841
Normalized Time           31.1158
========================  ==========

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.0909091
Weighted Fraction of Eqa  0.030303
Time (sec)                0.0513456
Normalized Time           1.61668
========================  =========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.349647
Weighted Fraction of Eqa   0.0554777
Time (sec)                 0.966773
Normalized Time           33.8824
========================  ==========

Random
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.153897
Weighted Fraction of Eqa  0.0311547
Time (sec)                0.28003
Normalized Time           7.51681
========================  =========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.732691
Weighted Fraction of Eqa    0.0718566
Time (sec)                  6.70744
Normalized Time           124.043
========================  ===========

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.464167
Weighted Fraction of Eqa  0.0838261
Time (sec)                0.0184558
Normalized Time           2.28596
========================  =========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.1
Time (sec)                0.000151873
Normalized Time           1
========================  ===========

Regret Matching
---------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Random                   0.924928             0.64514
Pure                     0.515728             0.236737
Biased                   0.10575              0.0276221
Role biased              0.102853             0.0250178
Uniform                  0.0637684            0.0137799
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Multiplicative Weights Stoch            0.927526      0.648969
Replicator Dynamics                     0.927218     80.6078
Fictitious Play Long                    0.926172      0.232047
Fictitious Play                         0.926072      8.98201
Multiplicative Weights Bandit           0.9153        1.17789
Multiplicative Weights Dist             0.904918      6.74234
Regret Minimization                     0.900291    559.022
Scarf 1                                 0.871919     84.6587
Scarf 5                                 0.871468     35.8871
Scarf 30                                0.871176     16.7646
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa                1
Weighted Fraction of Eqa       0.555556
Time (sec)                    33.8172
Normalized Time           228405
========================  =============

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.304548
Weighted Fraction of Eqa     0.034857
Time (sec)                 127.934
Normalized Time           7810.02
========================  ===========

Chicken
"""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0.0625
Weighted Fraction of Eqa    0.04
Time (sec)                  3.14211
Normalized Time           599.737
========================  =========

Zero sum
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.621495
Weighted Fraction of Eqa      0.391822
Time (sec)                  254.72
Normalized Time           19059.1
========================  ============

Mix
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                  2.96706
Normalized Time           538.77
========================  =========

Sineagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.1125
Time (sec)                 1.53544
Normalized Time           67.9379
========================  ========

Normagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.455498
Weighted Fraction of Eqa   0.0478788
Time (sec)                 4.64639
Normalized Time           72.936
========================  ==========

Polyagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa            0.0357143
Weighted Fraction of Eqa   0.00478316
Time (sec)                 2.82121
Normalized Time           77.3436
========================  ===========

Local effect
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.145833
Weighted Fraction of Eqa    0.0148201
Time (sec)                  5.05215
Normalized Time           176.661
========================  ===========

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.130333
Weighted Fraction of Eqa     0.0545
Time (sec)                  23.0914
Normalized Time           1201.27
========================  ===========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.385925
Weighted Fraction of Eqa     0.224672
Time (sec)                 243.786
Normalized Time           7511.41
========================  ===========

Shapley easy
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa                0.666667
Weighted Fraction of Eqa       0.366667
Time (sec)                    29.9445
Normalized Time           196860
========================  =============

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.525
Weighted Fraction of Eqa    0.0622159
Time (sec)                  0.693169
Normalized Time           116.204
========================  ===========

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.142857
Time (sec)                19.839
Normalized Time           17.2034
========================  =========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.305898
Weighted Fraction of Eqa   0.0395527
Time (sec)                 5.54689
Normalized Time           37.5924
========================  ==========

Hard
""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.0588235
Weighted Fraction of Eqa     0.0117647
Time (sec)                  78.2362
Normalized Time           5021
========================  ============

Gambit
""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.363636
Weighted Fraction of Eqa   0.0454545
Time (sec)                 1.52629
Normalized Time           48.0573
========================  ==========

Polymatrix
""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.405965
Weighted Fraction of Eqa     0.0664618
Time (sec)                  39.8172
Normalized Time           1395.47
========================  ============

Random
""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.261022
Weighted Fraction of Eqa     0.0723834
Time (sec)                  76.8243
Normalized Time           2062.19
========================  ============

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.760323
Weighted Fraction of Eqa   0.0714043
Time (sec)                 2.72771
Normalized Time           50.4443
========================  ==========

Roshambo
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.7075
Weighted Fraction of Eqa     0.323323
Time (sec)                  33.247
Normalized Time           4118.02
========================  ===========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                1
Weighted Fraction of Eqa       0.1
Time (sec)                    25.2879
Normalized Time           166507
========================  ===========

Multiplicative Weights Dist
---------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                  0.945395             0.845543
Biased                   0.163391             0.0462799
Role biased              0.160499             0.0431626
Random                   0.156742             0.0466468
Pure                     0.0609121            0.0172996
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Fictitious Play Long                    0.996554     0.0344164
Fictitious Play                         0.996255     1.33218
Replicator Dynamics                     0.994949    11.9555
Multiplicative Weights Stoch            0.978448     0.0962527
Multiplicative Weights Bandit           0.947091     0.1747
Regret Minimization                     0.94645     82.9121
Scarf 1                                 0.892497    12.5563
Scarf 5                                 0.892045     5.32264
Scarf 30                                0.891754     2.48646
Regret Matching                         0.565745     0.148316
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.5
Weighted Fraction of Eqa      0.0555556
Time (sec)                    2.92369
Normalized Time           19746.9
========================  =============

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.512942
Weighted Fraction of Eqa    0.160588
Time (sec)                  8.43969
Normalized Time           515.22
========================  ==========

Chicken
"""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.5375
Weighted Fraction of Eqa    0.0612103
Time (sec)                  2.60122
Normalized Time           496.497
========================  ===========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.373656
Weighted Fraction of Eqa    0.0858021
Time (sec)                  4.23339
Normalized Time           316.759
========================  ===========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.8
Weighted Fraction of Eqa    0.0832143
Time (sec)                  2.59525
Normalized Time           471.255
========================  ===========

Sineagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.1125
Time (sec)                 11.4917
Normalized Time           508.467
========================  ========

Normagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.655801
Weighted Fraction of Eqa    0.122854
Time (sec)                 32.0664
Normalized Time           503.358
========================  ==========

Polyagg large
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.285714
Weighted Fraction of Eqa      0.220557
Time (sec)                 1704.14
Normalized Time           46719.1
========================  ============

Local effect
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.576389
Weighted Fraction of Eqa    0.0670621
Time (sec)                 15.1364
Normalized Time           529.282
========================  ===========

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.529003
Weighted Fraction of Eqa    0.102581
Time (sec)                 16.2656
Normalized Time           846.174
========================  ==========

Covariant
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.337059
Weighted Fraction of Eqa    0.167468
Time (sec)                  5.92329
Normalized Time           182.506
========================  ==========

Shapley easy
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.333333
Weighted Fraction of Eqa      0.0333333
Time (sec)                    2.89428
Normalized Time           19027.4
========================  =============

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.558333
Weighted Fraction of Eqa    0.0822159
Time (sec)                  2.58478
Normalized Time           433.317
========================  ===========

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.142857
Time (sec)                69.3236
Normalized Time           60.1139
========================  =========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.437487
Weighted Fraction of Eqa    0.108665
Time (sec)                 33.4532
Normalized Time           226.719
========================  ==========

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.235294
Weighted Fraction of Eqa    0.188235
Time (sec)                  4.93074
Normalized Time           316.442
========================  ==========

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.454545
Weighted Fraction of Eqa    0.0636364
Time (sec)                  5.28843
Normalized Time           166.513
========================  ===========

Polymatrix
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.445319
Weighted Fraction of Eqa    0.0851808
Time (sec)                  5.67596
Normalized Time           198.925
========================  ===========

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.326832
Weighted Fraction of Eqa    0.0679519
Time (sec)                  4.73046
Normalized Time           126.979
========================  ===========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.796538
Weighted Fraction of Eqa    0.0885722
Time (sec)                 30.7712
Normalized Time           569.061
========================  ===========

Roshambo
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.380833
Weighted Fraction of Eqa    0.0529392
Time (sec)                  2.99483
Normalized Time           370.944
========================  ===========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa               1
Weighted Fraction of Eqa      0.1
Time (sec)                    2.88606
Normalized Time           19003.2
========================  ===========

Replicator Dynamics
-------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                  0.942031             0.850344
Biased                   0.143336             0.0352437
Role biased              0.143326             0.0355169
Random                   0.142065             0.0361951
Pure                     0.0752944            0.0219619
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Fictitious Play Long                    0.974041    0.00287872
Fictitious Play                         0.973601    0.111429
Multiplicative Weights Dist             0.954491    0.0836439
Multiplicative Weights Stoch            0.954236    0.00805094
Multiplicative Weights Bandit           0.942476    0.0146126
Regret Minimization                     0.941664    6.93509
Scarf 1                                 0.887903    1.05025
Scarf 5                                 0.887451    0.445206
Scarf 30                                0.88716     0.207977
Regret Matching                         0.556909    0.0124058
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.5
Weighted Fraction of Eqa    0.0555556
Time (sec)                  0.0477914
Normalized Time           322.788
========================  ===========

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.436103
Weighted Fraction of Eqa    0.0885598
Time (sec)                  8.00409
Normalized Time           488.628
========================  ===========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.5375
Weighted Fraction of Eqa   0.0612103
Time (sec)                 0.0729519
Normalized Time           13.9244
========================  ==========

Zero sum
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.120139
Weighted Fraction of Eqa   0.0142472
Time (sec)                 0.676059
Normalized Time           50.5854
========================  ==========

Mix
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.8
Weighted Fraction of Eqa  0.0832143
Time (sec)                0.0424278
Normalized Time           7.7042
========================  =========

Sineagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.1125
Time (sec)                0.219549
Normalized Time           9.7143
========================  ========

Normagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.594784
Weighted Fraction of Eqa   0.0688609
Time (sec)                 1.41217
Normalized Time           22.1674
========================  ==========

Polyagg large
"""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa              0.0535714
Weighted Fraction of Eqa     0.00924745
Time (sec)                  46.3598
Normalized Time           1270.96
========================  =============

Local effect
""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.576389
Weighted Fraction of Eqa   0.0670621
Time (sec)                 0.463779
Normalized Time           16.2172
========================  ==========

Congestion
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.538898
Weighted Fraction of Eqa   0.116055
Time (sec)                 1.25418
Normalized Time           65.2456
========================  =========

Covariant
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.161206
Weighted Fraction of Eqa   0.0185797
Time (sec)                 1.29125
Normalized Time           39.7854
========================  ==========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.333333
Weighted Fraction of Eqa    0.0333333
Time (sec)                  0.0478084
Normalized Time           314.299
========================  ===========

Prisoners
"""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.541667
Weighted Fraction of Eqa  0.0655492
Time (sec)                0.0424702
Normalized Time           7.11978
========================  =========

Normagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.142857
Time (sec)                1.69171
Normalized Time           1.46697
========================  ========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.365775
Weighted Fraction of Eqa   0.0514696
Time (sec)                 4.76914
Normalized Time           32.3215
========================  ==========

Hard
""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                  7.1365
Normalized Time           458.002
========================  ========

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.454545
Weighted Fraction of Eqa  0.0636364
Time (sec)                0.0791451
Normalized Time           2.49199
========================  =========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.351567
Weighted Fraction of Eqa   0.0363664
Time (sec)                 0.303221
Normalized Time           10.627
========================  ==========

Random
""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.29407
Weighted Fraction of Eqa   0.0453142
Time (sec)                 0.469837
Normalized Time           12.6118
========================  ==========

Polyagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.781501
Weighted Fraction of Eqa   0.075289
Time (sec)                 0.923911
Normalized Time           17.0862
========================  =========

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0
Weighted Fraction of Eqa  0
Time (sec)                0.0532223
Normalized Time           6.59218
========================  =========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.1
Time (sec)                  0.0481976
Normalized Time           317.356
========================  ===========

Fictitious Play
---------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                   0.968887            0.844597
Pure                      0.179452            0.0399854
Random                    0.177787            0.040904
Biased                    0.176497            0.0375596
Role biased               0.172362            0.0357947
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Fictitious Play Long                    0.996304     0.0258347
Replicator Dynamics                     0.985459     8.97436
Multiplicative Weights Stoch            0.970015     0.0722521
Multiplicative Weights Dist             0.965673     0.75065
Regret Minimization                     0.945646    62.238
Multiplicative Weights Bandit           0.94078      0.131139
Scarf 1                                 0.890577     9.42536
Scarf 5                                 0.890126     3.99544
Scarf 30                                0.889834     1.86646
Regret Matching                         0.564136     0.111334
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.5
Weighted Fraction of Eqa      0.0555556
Time (sec)                    3.42748
Normalized Time           23149.6
========================  =============

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.376006
Weighted Fraction of Eqa    0.0486926
Time (sec)                  7.0972
Normalized Time           433.264
========================  ===========

Chicken
"""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.5375
Weighted Fraction of Eqa    0.0612103
Time (sec)                  3.16318
Normalized Time           603.759
========================  ===========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.345037
Weighted Fraction of Eqa    0.0625309
Time (sec)                  4.30669
Normalized Time           322.244
========================  ===========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.8
Weighted Fraction of Eqa    0.0832143
Time (sec)                  3.1759
Normalized Time           576.692
========================  ===========

Sineagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.1125
Time (sec)                 15.0365
Normalized Time           665.315
========================  ========

Normagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.575141
Weighted Fraction of Eqa    0.0639502
Time (sec)                 10.5182
Normalized Time           165.108
========================  ===========

Polyagg large
"""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa              0.0357143
Weighted Fraction of Eqa     0.00478316
Time (sec)                 204.126
Normalized Time           5596.11
========================  =============

Local effect
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.576389
Weighted Fraction of Eqa    0.0670621
Time (sec)                 15.187
Normalized Time           531.051
========================  ===========

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.453237
Weighted Fraction of Eqa    0.0506665
Time (sec)                 18.3005
Normalized Time           952.033
========================  ===========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.193276
Weighted Fraction of Eqa    0.0357827
Time (sec)                  4.168
Normalized Time           128.422
========================  ===========

Shapley easy
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.333333
Weighted Fraction of Eqa      0.0333333
Time (sec)                    3.4285
Normalized Time           22539.4
========================  =============

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.525
Weighted Fraction of Eqa   0.0622159
Time (sec)                 0.289863
Normalized Time           48.5931
========================  ==========

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.142857
Time (sec)                70.5892
Normalized Time           61.2113
========================  =========

Sineagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.35141
Weighted Fraction of Eqa    0.0470733
Time (sec)                 16.5122
Normalized Time           111.907
========================  ===========

Hard
""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.0588235
Weighted Fraction of Eqa    0.0117647
Time (sec)                  5.68838
Normalized Time           365.066
========================  ===========

Gambit
""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.454545
Weighted Fraction of Eqa   0.0636364
Time (sec)                 1.13567
Normalized Time           35.7579
========================  ==========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.385215
Weighted Fraction of Eqa   0.0425868
Time (sec)                 1.50801
Normalized Time           52.8512
========================  ==========

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.315737
Weighted Fraction of Eqa    0.0726808
Time (sec)                  3.85757
Normalized Time           103.548
========================  ===========

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.774358
Weighted Fraction of Eqa   0.0788605
Time (sec)                 4.29832
Normalized Time           79.4902
========================  ==========

Roshambo
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.380833
Weighted Fraction of Eqa    0.0529392
Time (sec)                  3.59557
Normalized Time           445.351
========================  ===========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa               1
Weighted Fraction of Eqa      0.1
Time (sec)                    3.43402
Normalized Time           22611.2
========================  ===========

Fictitious Play Long
--------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                   0.968629            0.84456
Pure                      0.178594            0.0403717
Biased                    0.176104            0.037672
Random                    0.173102            0.0400963
Role biased               0.172011            0.0359403
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Fictitious Play                         0.997874      38.7077
Replicator Dynamics                     0.985459     347.377
Multiplicative Weights Stoch            0.969578       2.79671
Multiplicative Weights Dist             0.965337      29.0559
Regret Minimization                     0.944251    2409.09
Multiplicative Weights Bandit           0.940954       5.07608
Scarf 1                                 0.89096      364.834
Scarf 5                                 0.890508     154.654
Scarf 30                                0.890217      72.2465
Regret Matching                         0.564378       4.30947
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa             0.5
Weighted Fraction of Eqa    0.0555556
Time (sec)                168.961
Normalized Time             1.14118e+06
========================  =============

Rbf
"""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.376602
Weighted Fraction of Eqa      0.0492878
Time (sec)                  173.48
Normalized Time           10590.5
========================  =============

Chicken
"""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.5375
Weighted Fraction of Eqa      0.0612103
Time (sec)                   73.9022
Normalized Time           14105.8
========================  =============

Zero sum
""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.323985
Weighted Fraction of Eqa      0.0460835
Time (sec)                  134.481
Normalized Time           10062.4
========================  =============

Mix
"""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.8
Weighted Fraction of Eqa      0.0832143
Time (sec)                   87.0887
Normalized Time           15813.9
========================  =============

Sineagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.1125
Time (sec)                 152.084
Normalized Time           6729.21
========================  =========

Normagg small
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.575141
Weighted Fraction of Eqa     0.0639502
Time (sec)                 150.813
Normalized Time           2367.37
========================  ============

Polyagg large
"""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa              0.0357143
Weighted Fraction of Eqa     0.00478316
Time (sec)                 313.31
Normalized Time           8589.39
========================  =============

Local effect
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.576389
Weighted Fraction of Eqa      0.0670621
Time (sec)                  489.299
Normalized Time           17109.5
========================  =============

Congestion
""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.453237
Weighted Fraction of Eqa      0.0506665
Time (sec)                  639.256
Normalized Time           33255.6
========================  =============

Covariant
"""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.179163
Weighted Fraction of Eqa     0.0236679
Time (sec)                 104.171
Normalized Time           3209.66
========================  ============

Shapley easy
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa             0.333333
Weighted Fraction of Eqa    0.0333333
Time (sec)                199.545
Normalized Time             1.31183e+06
========================  =============

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.525
Weighted Fraction of Eqa   0.0622159
Time (sec)                 0.290636
Normalized Time           48.7227
========================  ==========

Normagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.142857
Time (sec)                1886.26
Normalized Time           1635.67
========================  ===========

Sineagg small
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.34766
Weighted Fraction of Eqa     0.0433233
Time (sec)                 266.383
Normalized Time           1805.34
========================  ============

Hard
""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.0588235
Weighted Fraction of Eqa      0.0117647
Time (sec)                  186.185
Normalized Time           11948.9
========================  =============

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.454545
Weighted Fraction of Eqa    0.0636364
Time (sec)                 25.2319
Normalized Time           794.46
========================  ===========

Polymatrix
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.385215
Weighted Fraction of Eqa    0.0425868
Time (sec)                 22.0073
Normalized Time           771.287
========================  ===========

Random
""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.291578
Weighted Fraction of Eqa     0.0485217
Time (sec)                 102.072
Normalized Time           2739.9
========================  ============

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.774358
Weighted Fraction of Eqa   0.0788605
Time (sec)                 4.52731
Normalized Time           83.7249
========================  ==========

Roshambo
""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.355833
Weighted Fraction of Eqa      0.0493677
Time (sec)                  182.412
Normalized Time           22593.7
========================  =============

Shapley normal
""""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa             1
Weighted Fraction of Eqa    0.1
Time (sec)                190.732
Normalized Time             1.25587e+06
========================  =============

Multiplicative Weights Stoch
----------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Random                   0.159355             0.0463557
Biased                   0.157019             0.0422164
Role biased              0.152887             0.0409085
Uniform                  0.123094             0.0294856
Pure                     0.0740799            0.0203774
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Multiplicative Weights Bandit          0.9531         1.81502
Fictitious Play Long                   0.172797       0.357563
Fictitious Play                        0.17243       13.8404
Replicator Dynamics                    0.168424     124.209
Regret Matching                        0.159336       1.54091
Multiplicative Weights Dist            0.146658      10.3893
Regret Minimization                    0.128161     861.401
Scarf 1                                0.0885782    130.451
Scarf 5                                0.0881266     55.2986
Scarf 30                               0.087835      25.8327
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                0
Weighted Fraction of Eqa       0
Time (sec)                    85.1506
Normalized Time           575117
========================  ===========

Rbf
"""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.335488
Weighted Fraction of Eqa      0.0522358
Time (sec)                  197.324
Normalized Time           12046.1
========================  =============

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.8125
Weighted Fraction of Eqa    0.246419
Time (sec)                  0.739706
Normalized Time           141.188
========================  ==========

Zero sum
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.329394
Weighted Fraction of Eqa      0.047297
Time (sec)                  197.321
Normalized Time           14764.3
========================  ============

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.95
Weighted Fraction of Eqa    0.214464
Time (sec)                  0.726422
Normalized Time           131.906
========================  ==========

Sineagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.1125
Time (sec)                  2.76868
Normalized Time           122.505
========================  =========

Normagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.593593
Weighted Fraction of Eqa    0.0710435
Time (sec)                 15.3826
Normalized Time           241.466
========================  ===========

Polyagg large
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.0892857
Weighted Fraction of Eqa     0.0241284
Time (sec)                  73.1534
Normalized Time           2005.5
========================  ============

Local effect
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.576389
Weighted Fraction of Eqa    0.0670621
Time (sec)                  4.79695
Normalized Time           167.737
========================  ===========

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.559266
Weighted Fraction of Eqa     0.119028
Time (sec)                  24.1546
Normalized Time           1256.58
========================  ===========

Covariant
"""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.221167
Weighted Fraction of Eqa     0.0503129
Time (sec)                 214.367
Normalized Time           6604.98
========================  ============

Shapley easy
""""""""""""

========================  ==============
Metric                             Value
========================  ==============
Fraction of Eqa                0.333333
Weighted Fraction of Eqa       0.0333333
Time (sec)                    27.8162
Normalized Time           182868
========================  ==============

Prisoners
"""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.541667
Weighted Fraction of Eqa     0.0655492
Time (sec)                   6.17225
Normalized Time           1034.73
========================  ============

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.142857
Time (sec)                10.0131
Normalized Time            8.68284
========================  =========

Sineagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.37734
Weighted Fraction of Eqa    0.0736149
Time (sec)                 21.801
Normalized Time           147.75
========================  ===========

Hard
""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.0588235
Weighted Fraction of Eqa     0.0117647
Time (sec)                  91.2241
Normalized Time           5854.53
========================  ============

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.363636
Weighted Fraction of Eqa    0.0454545
Time (sec)                 14.5183
Normalized Time           457.126
========================  ===========

Polymatrix
""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.389067
Weighted Fraction of Eqa     0.0434497
Time (sec)                  69.5512
Normalized Time           2437.56
========================  ============

Random
""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.280575
Weighted Fraction of Eqa     0.0560627
Time (sec)                  56.8294
Normalized Time           1525.47
========================  ============

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.781501
Weighted Fraction of Eqa    0.075289
Time (sec)                 11.4221
Normalized Time           211.233
========================  ==========

Roshambo
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.380833
Weighted Fraction of Eqa     0.0529392
Time (sec)                  60.4415
Normalized Time           7486.36
========================  ============

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                1
Weighted Fraction of Eqa       0.1
Time (sec)                    86.3627
Normalized Time           568652
========================  ===========

Scarf 1
-------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                    0.99136              0.99136
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Scarf 5                                 0.997329    0.423903
Scarf 30                                0.996585    0.198026
Replicator Dynamics                     0.891199    0.95215
Multiplicative Weights Bandit           0.887203    0.0139134
Multiplicative Weights Stoch            0.882973    0.00766571
Regret Minimization                     0.877238    6.60325
Fictitious Play                         0.875705    0.106097
Fictitious Play Long                    0.875311    0.00274097
Multiplicative Weights Dist             0.875229    0.0796415
Regret Matching                         0.49575     0.0118121
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.5
Weighted Fraction of Eqa  0.0555556
Time (sec)                0.000176191
Normalized Time           1.19002
========================  ===========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.318822
Weighted Fraction of Eqa    0.073497
Time (sec)                  3.64492
Normalized Time           222.512
========================  ==========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.5375
Weighted Fraction of Eqa  0.0772222
Time (sec)                0.00536712
Normalized Time           1.02443
========================  ==========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.312472
Weighted Fraction of Eqa    0.0581924
Time (sec)                  1.6169
Normalized Time           120.983
========================  ===========

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.8
Weighted Fraction of Eqa  0.0954167
Time (sec)                0.00701667
Normalized Time           1.27411
========================  ==========

Sineagg large
"""""""""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa            0.5
Weighted Fraction of Eqa   0.05
Time (sec)                 1.5415
Normalized Time           68.206
========================  =======

Normagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.46131
Weighted Fraction of Eqa   0.060965
Time (sec)                 6.22083
Normalized Time           97.6506
========================  =========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.0178571
Weighted Fraction of Eqa  0.00595238
Time (sec)                0.0366912
Normalized Time           1.00589
========================  ==========

Local effect
""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.576389
Weighted Fraction of Eqa  0.106637
Time (sec)                0.189967
Normalized Time           6.64268
========================  ========

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.453237
Weighted Fraction of Eqa   0.0686254
Time (sec)                 1.63158
Normalized Time           84.8785
========================  ==========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.153815
Weighted Fraction of Eqa    0.0336958
Time (sec)                  7.43034
Normalized Time           228.94
========================  ===========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.333333
Weighted Fraction of Eqa  0.0333333
Time (sec)                0.000173092
Normalized Time           1.13793
========================  ===========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.525
Weighted Fraction of Eqa  0.148674
Time (sec)                0.00917681
Normalized Time           1.53842
========================  ==========

Normagg large
"""""""""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa            0
Weighted Fraction of Eqa   0
Time (sec)                60.0021
Normalized Time           52.0307
========================  =======

Sineagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.239765
Weighted Fraction of Eqa  0.0375492
Time (sec)                0.687618
Normalized Time           4.66013
========================  =========

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.0588235
Weighted Fraction of Eqa   0.0196078
Time (sec)                 0.482798
Normalized Time           30.9847
========================  ==========

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.0909091
Weighted Fraction of Eqa  0.030303
Time (sec)                0.0517638
Normalized Time           1.62985
========================  =========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.349647
Weighted Fraction of Eqa   0.0554777
Time (sec)                 0.965104
Normalized Time           33.8239
========================  ==========

Random
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.153897
Weighted Fraction of Eqa  0.0311547
Time (sec)                0.28036
Normalized Time           7.52566
========================  =========

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.725548
Weighted Fraction of Eqa   0.0711423
Time (sec)                 3.22001
Normalized Time           59.5486
========================  ==========

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.464167
Weighted Fraction of Eqa  0.0838261
Time (sec)                0.0186894
Normalized Time           2.31489
========================  =========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.1
Time (sec)                0.000180483
Normalized Time           1.18838
========================  ===========

Multiplicative Weights Bandit
-----------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Random                   0.115048             0.0483717
Biased                   0.111203             0.038484
Role biased              0.106899             0.035918
Uniform                  0.0676234            0.0180948
Pure                     0.0614487            0.0178271
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

============================  =================  ============
Method                          Fraction of Eqa    Time Ratio
============================  =================  ============
Multiplicative Weights Stoch          0.949729       0.550958
Regret Matching                       0.148754       0.848975
Replicator Dynamics                   0.14488       68.434
Fictitious Play                       0.12992        7.6255
Fictitious Play Long                  0.128606       0.197002
Multiplicative Weights Dist           0.112712       5.72408
Regret Minimization                   0.108446     474.596
Scarf 1                               0.0709794     71.8731
Scarf 5                               0.0705277     30.4673
Scarf 30                              0.0702361     14.2327
============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                0
Weighted Fraction of Eqa       0
Time (sec)                    37.7035
Normalized Time           254653
========================  ===========

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.433037
Weighted Fraction of Eqa     0.141674
Time (sec)                  45.2721
Normalized Time           2763.74
========================  ===========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.620833
Weighted Fraction of Eqa    0.175863
Time (sec)                  5.04106
Normalized Time           962.191
========================  ==========

Zero sum
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.207575
Weighted Fraction of Eqa     0.0814147
Time (sec)                  61.3869
Normalized Time           4593.21
========================  ============

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.8
Weighted Fraction of Eqa    0.0832143
Time (sec)                  2.10556
Normalized Time           382.334
========================  ===========

Sineagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.1125
Time (sec)                 1.31277
Normalized Time           58.0854
========================  ========

Normagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.611212
Weighted Fraction of Eqa    0.169466
Time (sec)                 16.1931
Normalized Time           254.188
========================  ==========

Polyagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.125
Weighted Fraction of Eqa    0.0509141
Time (sec)                 30.0901
Normalized Time           824.921
========================  ===========

Local effect
""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.704167
Weighted Fraction of Eqa    0.167062
Time (sec)                  7.22445
Normalized Time           252.621
========================  ==========

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.617381
Weighted Fraction of Eqa     0.21696
Time (sec)                  20.0386
Normalized Time           1042.45
========================  ===========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.406794
Weighted Fraction of Eqa     0.246532
Time (sec)                  75.293
Normalized Time           2319.89
========================  ===========

Shapley easy
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa                0.333333
Weighted Fraction of Eqa       0.333333
Time (sec)                    42.2664
Normalized Time           277865
========================  =============

Prisoners
"""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.566667
Weighted Fraction of Eqa     0.0717992
Time (sec)                   6.16585
Normalized Time           1033.65
========================  ============

Normagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.142857
Time (sec)                3.99676
Normalized Time           3.46578
========================  ========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.434973
Weighted Fraction of Eqa    0.152492
Time (sec)                 15.3441
Normalized Time           103.99
========================  ==========

Hard
""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.411765
Weighted Fraction of Eqa     0.411765
Time (sec)                  63.0848
Normalized Time           4048.62
========================  ===========

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.363636
Weighted Fraction of Eqa    0.0454545
Time (sec)                  6.41273
Normalized Time           201.913
========================  ===========

Polymatrix
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.725753
Weighted Fraction of Eqa     0.354831
Time (sec)                  64.5869
Normalized Time           2263.57
========================  ===========

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.28421
Weighted Fraction of Eqa     0.108513
Time (sec)                  41.0832
Normalized Time           1102.79
========================  ===========

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.781501
Weighted Fraction of Eqa    0.075289
Time (sec)                 12.4333
Normalized Time           229.932
========================  ==========

Roshambo
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.26
Weighted Fraction of Eqa     0.116534
Time (sec)                  41.2253
Normalized Time           5106.21
========================  ===========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                0
Weighted Fraction of Eqa       0
Time (sec)                    42.6458
Normalized Time           280800
========================  ===========

Regret Minimization
-------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.913189            0.238037
Random                    0.909974            0.238086
Role biased               0.905214            0.231136
Uniform                   0.89146             0.222137
Pure                      0.105826            0.0560536
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Replicator Dynamics                     0.964337   0.144194
Fictitious Play                         0.954297   0.0160674
Fictitious Play Long                    0.953767   0.000415095
Multiplicative Weights Stoch            0.938871   0.0011609
Multiplicative Weights Dist             0.937148   0.012061
Multiplicative Weights Bandit           0.929212   0.00210706
Scarf 1                                 0.879939   0.151441
Scarf 5                                 0.879487   0.0641962
Scarf 30                                0.879196   0.0299891
Regret Matching                         0.543883   0.00178884
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.5
Weighted Fraction of Eqa   0.0555556
Time (sec)                 0.0068413
Normalized Time           46.2069
========================  ==========

Rbf
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.520615
Weighted Fraction of Eqa  0.201114
Time (sec)                0.0163808
Normalized Time           1
========================  =========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.5375
Weighted Fraction of Eqa  0.0612103
Time (sec)                0.00654014
Normalized Time           1.24832
========================  ==========

Zero sum
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.276208
Weighted Fraction of Eqa  0.0962257
Time (sec)                0.0133647
Normalized Time           1
========================  =========

Mix
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.8
Weighted Fraction of Eqa  0.0832143
Time (sec)                0.0055071
Normalized Time           1
========================  =========

Sineagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.5
Weighted Fraction of Eqa  0.0625
Time (sec)                0.0226006
Normalized Time           1
========================  =========

Normagg small
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.658149
Weighted Fraction of Eqa  0.205024
Time (sec)                0.063705
Normalized Time           1
========================  ========

Polyagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.678571
Weighted Fraction of Eqa     0.662946
Time (sec)                  82.0281
Normalized Time           2248.8
========================  ===========

Local effect
""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.613889
Weighted Fraction of Eqa  0.162895
Time (sec)                0.028598
Normalized Time           1
========================  ========

Congestion
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.456218
Weighted Fraction of Eqa  0.0836675
Time (sec)                0.0192225
Normalized Time           1
========================  =========

Covariant
"""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.283948
Weighted Fraction of Eqa  0.130335
Time (sec)                0.0324554
Normalized Time           1
========================  =========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa            0.333333
Weighted Fraction of Eqa   0.0333333
Time (sec)                 0.00646641
Normalized Time           42.5111
========================  ===========

Prisoners
"""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.558333
Weighted Fraction of Eqa  0.0822159
Time (sec)                0.0059651
Normalized Time           1
========================  =========

Normagg large
"""""""""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa           0
Weighted Fraction of Eqa  0
Time (sec)                1.15321
Normalized Time           1
========================  =======

Sineagg small
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.674401
Weighted Fraction of Eqa  0.371161
Time (sec)                0.147553
Normalized Time           1
========================  ========

Hard
""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.294118
Weighted Fraction of Eqa  0.294118
Time (sec)                0.0155818
Normalized Time           1
========================  =========

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.909091
Weighted Fraction of Eqa  0.518182
Time (sec)                0.0317599
Normalized Time           1
========================  =========

Polymatrix
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.502937
Weighted Fraction of Eqa  0.162104
Time (sec)                0.0285332
Normalized Time           1
========================  =========

Random
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.64973
Weighted Fraction of Eqa  0.435108
Time (sec)                0.0372538
Normalized Time           1
========================  =========

Polyagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.949546
Weighted Fraction of Eqa  0.24158
Time (sec)                0.0540736
Normalized Time           1
========================  =========

Roshambo
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.4975
Weighted Fraction of Eqa  0.100479
Time (sec)                0.00807355
Normalized Time           1
========================  ==========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.1
Time (sec)                 0.00644888
Normalized Time           42.4624
========================  ===========

Scarf 5
-------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                   0.994031             0.994031
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Scarf 1                                 1           2.35903
Scarf 30                                0.999257    0.467148
Replicator Dynamics                     0.89197     2.24615
Multiplicative Weights Bandit           0.88933     0.0328221
Multiplicative Weights Stoch            0.8851      0.0180836
Regret Minimization                     0.879302   15.5772
Fictitious Play                         0.877925    0.250285
Fictitious Play Long                    0.877531    0.00646603
Multiplicative Weights Dist             0.875799    0.187877
Regret Matching                         0.49797     0.0278652
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.5
Weighted Fraction of Eqa  0.0555556
Time (sec)                0.000164747
Normalized Time           1.11272
========================  ===========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.331322
Weighted Fraction of Eqa    0.074747
Time (sec)                 10.8419
Normalized Time           661.87
========================  ==========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.5375
Weighted Fraction of Eqa  0.0772222
Time (sec)                0.00525931
Normalized Time           1.00385
========================  ==========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.312472
Weighted Fraction of Eqa    0.0581924
Time (sec)                  1.6158
Normalized Time           120.9
========================  ===========

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.8
Weighted Fraction of Eqa  0.0954167
Time (sec)                0.00688252
Normalized Time           1.24975
========================  ==========

Sineagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa            0.5
Weighted Fraction of Eqa   0.05
Time (sec)                 1.53476
Normalized Time           67.9077
========================  ========

Normagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.464881
Weighted Fraction of Eqa    0.0627507
Time (sec)                 26.1881
Normalized Time           411.084
========================  ===========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.0178571
Weighted Fraction of Eqa  0.00595238
Time (sec)                0.0364764
Normalized Time           1
========================  ==========

Local effect
""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.576389
Weighted Fraction of Eqa  0.106637
Time (sec)                0.189879
Normalized Time           6.63957
========================  ========

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.453237
Weighted Fraction of Eqa   0.0686254
Time (sec)                 1.63183
Normalized Time           84.8917
========================  ==========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.153815
Weighted Fraction of Eqa    0.0336958
Time (sec)                 19.4318
Normalized Time           598.722
========================  ===========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.333333
Weighted Fraction of Eqa  0.0333333
Time (sec)                0.000152349
Normalized Time           1.00157
========================  ===========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.525
Weighted Fraction of Eqa  0.148674
Time (sec)                0.00904394
Normalized Time           1.51614
========================  ==========

Normagg large
"""""""""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                300.002
Normalized Time           260.146
========================  =======

Sineagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.239765
Weighted Fraction of Eqa  0.0375492
Time (sec)                0.687357
Normalized Time           4.65837
========================  =========

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.0588235
Weighted Fraction of Eqa   0.0196078
Time (sec)                 0.501494
Normalized Time           32.1846
========================  ==========

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.0909091
Weighted Fraction of Eqa  0.030303
Time (sec)                0.0513351
Normalized Time           1.61635
========================  =========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.349647
Weighted Fraction of Eqa   0.0554777
Time (sec)                 0.965855
Normalized Time           33.8502
========================  ==========

Random
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.153897
Weighted Fraction of Eqa  0.0311547
Time (sec)                0.2803
Normalized Time           7.52407
========================  =========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.732691
Weighted Fraction of Eqa    0.0718566
Time (sec)                  6.71802
Normalized Time           124.238
========================  ===========

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.464167
Weighted Fraction of Eqa  0.0838261
Time (sec)                0.0185103
Normalized Time           2.29271
========================  =========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.1
Time (sec)                0.000156879
Normalized Time           1.03297
========================  ===========

