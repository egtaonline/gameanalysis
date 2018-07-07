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
Multiplicative Weights Dist             0.573797            0.112644      20.9828            972.496
Regret Minimization                     0.56805             0.161535       0.367683            3.23681
Multiplicative Weights Stoch            0.525651            0.0860005     51.4117          19734.8
Multiplicative Weights Bandit           0.52149             0.140106      26.4581          11952
Fictitious Play                         0.507327            0.0622499      9.55377           930.691
Fictitious Play Long                    0.505039            0.0592274    215.366           42578.1
Replicator Dynamics                     0.491034            0.0653753      1.81025            98.6114
Scarf 30                                0.462414            0.0720758     10.6574             63.3012
Scarf 5                                 0.461428            0.0710896      6.86547            61.5425
Scarf 1                                 0.457154            0.0706622      2.66938            38.9245
Regret Matching                         0.41264             0.0990347     45.5782          11860.9
=============================  =================  ===================  ============  =================

Regret Matching
---------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Random                    0.809317            0.224396
Role biased               0.767526            0.196218
Biased                    0.747552            0.179043
Pure                      0.733524            0.188962
Uniform                   0.376268            0.0785053
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Multiplicative Weights Stoch            0.817922      0.601013
Multiplicative Weights Bandit           0.797347      0.992377
Replicator Dynamics                     0.774885    120.279
Fictitious Play Long                    0.731701      0.278568
Fictitious Play                         0.730115     12.7442
Multiplicative Weights Dist             0.698985     12.1963
Regret Minimization                     0.539261   3664.38
Scarf 5                                 0.294111    192.727
Scarf 1                                 0.294111    304.716
Scarf 30                                0.278884    187.372
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.363636
Weighted Fraction of Eqa   0.0454545
Time (sec)                 2.49715
Normalized Time           61.7903
========================  ==========

Local effect
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.7
Weighted Fraction of Eqa    0.0636364
Time (sec)                 12.5793
Normalized Time           535.947
========================  ===========

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.733333
Weighted Fraction of Eqa   0.0945455
Time (sec)                 3.89227
Normalized Time           73.5962
========================  ==========

Roshambo
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.75
Weighted Fraction of Eqa     0.418333
Time (sec)                  37.5202
Normalized Time           4618.09
========================  ===========

Shapley easy
""""""""""""

========================  ==============
Metric                             Value
========================  ==============
Fraction of Eqa                0.5
Weighted Fraction of Eqa       0.0454545
Time (sec)                    34.4631
Normalized Time           226566
========================  ==============

Hard
""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa              0
Weighted Fraction of Eqa     0
Time (sec)                  57.729
Normalized Time           4081.28
========================  ========

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.6
Weighted Fraction of Eqa    0.0681818
Time (sec)                  0.672901
Normalized Time           117.954
========================  ===========

Chicken
"""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0.05
Weighted Fraction of Eqa    0.01
Time (sec)                  0.65612
Normalized Time           160.341
========================  =========

Random
""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.376938
Weighted Fraction of Eqa     0.0615971
Time (sec)                  79.2302
Normalized Time           1642.96
========================  ============

Polymatrix
""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.438824
Weighted Fraction of Eqa     0.0800914
Time (sec)                 115.018
Normalized Time           3580.92
========================  ============

Normagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.55
Weighted Fraction of Eqa   0.0769886
Time (sec)                 8.18543
Normalized Time           54.0641
========================  ==========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                1
Weighted Fraction of Eqa       0.1
Time (sec)                    34.6563
Normalized Time           228193
========================  ===========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.0952381
Weighted Fraction of Eqa   0.0119048
Time (sec)                 6.41493
Normalized Time           33.7778
========================  ==========

Shapley hard
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa                1
Weighted Fraction of Eqa       0.111111
Time (sec)                    45.393
Normalized Time           302209
========================  =============

Zero sum
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.367857
Weighted Fraction of Eqa      0.228849
Time (sec)                  294.937
Normalized Time           16360.7
========================  ============

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.0909091
Time (sec)                1.48918
Normalized Time           1.00837
========================  =========

Polyagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0.25
Weighted Fraction of Eqa    0.03125
Time (sec)                  1.00848
Normalized Time           104.996
========================  =========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.366106
Weighted Fraction of Eqa     0.221543
Time (sec)                  71.0602
Normalized Time           2494.05
========================  ===========

Sineagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.230769
Weighted Fraction of Eqa   0.0384615
Time (sec)                41.5506
Normalized Time           19.552
========================  ==========

Mix
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                  4.56819
Normalized Time           761.57
========================  =========

Rbf
"""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.473333
Weighted Fraction of Eqa     0.0500758
Time (sec)                  18.354
Normalized Time           1850.66
========================  ============

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.0666667
Weighted Fraction of Eqa    0.0666667
Time (sec)                 14.1735
Normalized Time           497.918
========================  ===========

Replicator Dynamics
-------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.711407             0.187901
Role biased               0.707487             0.201327
Random                    0.62123              0.161408
Pure                      0.573825             0.1583
Uniform                   0.49272              0.192708
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Fictitious Play Long                    0.913189    0.00231601
Regret Matching                         0.912583    0.008314
Fictitious Play                         0.911095    0.105955
Multiplicative Weights Dist             0.880815    0.1014
Multiplicative Weights Stoch            0.871171    0.00499682
Multiplicative Weights Bandit           0.813337    0.00825062
Regret Minimization                     0.631773   30.4657
Scarf 1                                 0.274782    2.5334
Scarf 5                                 0.274782    1.60233
Scarf 30                                0.259554    1.55781
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.454545
Weighted Fraction of Eqa  0.0636364
Time (sec)                0.0788597
Normalized Time           1.95133
========================  =========

Local effect
""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.766667
Weighted Fraction of Eqa   0.0747475
Time (sec)                 0.335619
Normalized Time           14.2992
========================  ==========

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.866667
Weighted Fraction of Eqa   0.0862121
Time (sec)                 0.778469
Normalized Time           14.7195
========================  ==========

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0
Weighted Fraction of Eqa  0
Time (sec)                0.0504748
Normalized Time           6.21259
========================  =========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.5
Weighted Fraction of Eqa    0.0454545
Time (sec)                  0.0493644
Normalized Time           324.529
========================  ===========

Hard
""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                  7.61081
Normalized Time           538.063
========================  =========

Prisoners
"""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.6
Weighted Fraction of Eqa  0.0681818
Time (sec)                0.0426393
Normalized Time           7.47433
========================  =========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.616667
Weighted Fraction of Eqa   0.0700794
Time (sec)                 0.0427445
Normalized Time           10.4458
========================  ==========

Random
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.469574
Weighted Fraction of Eqa  0.0618135
Time (sec)                0.229932
Normalized Time           4.768
========================  =========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.393333
Weighted Fraction of Eqa   0.0411894
Time (sec)                 1.6497
Normalized Time           51.3611
========================  ==========

Normagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.530128
Weighted Fraction of Eqa  0.0532707
Time (sec)                1.46415
Normalized Time           9.67058
========================  =========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.1
Time (sec)                  0.0507351
Normalized Time           334.064
========================  ===========

Sineagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.335238
Weighted Fraction of Eqa  0.0542381
Time (sec)                0.95338
Normalized Time           5.02002
========================  =========

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.111111
Time (sec)                  0.0482204
Normalized Time           321.034
========================  ===========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa            0.0333333
Weighted Fraction of Eqa   0.00416667
Time (sec)                 0.258722
Normalized Time           14.3518
========================  ===========

Normagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.0909091
Time (sec)                40.2285
Normalized Time           27.2399
========================  ==========

Polyagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.5
Weighted Fraction of Eqa   0.08125
Time (sec)                 0.473058
Normalized Time           49.2516
========================  =========

Covariant
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.122479
Weighted Fraction of Eqa   0.0149088
Time (sec)                 0.377445
Normalized Time           13.2475
========================  ==========

Sineagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.692308
Weighted Fraction of Eqa  0.239744
Time (sec)                7.83925
Normalized Time           3.68882
========================  ========

Mix
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.766667
Weighted Fraction of Eqa  0.0795238
Time (sec)                0.0426546
Normalized Time           7.11101
========================  =========

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.613333
Weighted Fraction of Eqa     0.140076
Time (sec)                  10.3873
Normalized Time           1047.36
========================  ===========

Congestion
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.516667
Weighted Fraction of Eqa   0.125026
Time (sec)                 0.350988
Normalized Time           12.3302
========================  =========

Fictitious Play
---------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Pure                      0.948277             0.24945
Biased                    0.943649             0.241685
Random                    0.848899             0.196789
Role biased               0.760822             0.165447
Uniform                   0.576913             0.131023
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Fictitious Play Long                    0.991801     0.0218584
Replicator Dynamics                     0.97294      9.43796
Multiplicative Weights Stoch            0.931006     0.0471598
Regret Matching                         0.920961     0.0784672
Multiplicative Weights Dist             0.908853     0.957013
Multiplicative Weights Bandit           0.801212     0.077869
Regret Minimization                     0.631996   287.534
Scarf 5                                 0.290388    15.1227
Scarf 1                                 0.290388    23.9102
Scarf 30                                0.27516     14.7026
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.454545
Weighted Fraction of Eqa   0.0636364
Time (sec)                 1.76562
Normalized Time           43.689
========================  ==========

Local effect
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.766667
Weighted Fraction of Eqa    0.0747475
Time (sec)                  9.9755
Normalized Time           425.012
========================  ===========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.866667
Weighted Fraction of Eqa    0.0862121
Time (sec)                 19.2675
Normalized Time           364.316
========================  ===========

Roshambo
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.333333
Weighted Fraction of Eqa    0.0605556
Time (sec)                  3.58973
Normalized Time           441.835
========================  ===========

Shapley easy
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.5
Weighted Fraction of Eqa      0.0454545
Time (sec)                    2.57354
Normalized Time           16918.8
========================  =============

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.0625
Weighted Fraction of Eqa    0.015625
Time (sec)                  5.72201
Normalized Time           404.53
========================  ==========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.6
Weighted Fraction of Eqa   0.0681818
Time (sec)                 0.289536
Normalized Time           50.7535
========================  ==========

Chicken
"""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.616667
Weighted Fraction of Eqa    0.0700794
Time (sec)                  3.17014
Normalized Time           774.711
========================  ===========

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.469574
Weighted Fraction of Eqa    0.0664646
Time (sec)                  4.90195
Normalized Time           101.65
========================  ===========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.403529
Weighted Fraction of Eqa   0.0447972
Time (sec)                 2.01369
Normalized Time           62.6932
========================  ==========

Normagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.530128
Weighted Fraction of Eqa    0.0532707
Time (sec)                 22.3093
Normalized Time           147.351
========================  ===========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa               1
Weighted Fraction of Eqa      0.1
Time (sec)                    2.58092
Normalized Time           16994
========================  ===========

Sineagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.315238
Weighted Fraction of Eqa    0.0502381
Time (sec)                 28.5793
Normalized Time           150.484
========================  ===========

Shapley hard
""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               1
Weighted Fraction of Eqa      0.111111
Time (sec)                    2.55201
Normalized Time           16990.3
========================  ============

Zero sum
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.253571
Weighted Fraction of Eqa    0.078254
Time (sec)                  4.19312
Normalized Time           232.6
========================  ==========

Normagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.0909091
Time (sec)                57.3253
Normalized Time           38.8166
========================  ==========

Polyagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0.25
Weighted Fraction of Eqa    0.03125
Time (sec)                  4.48278
Normalized Time           466.716
========================  =========

Covariant
"""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.160812
Weighted Fraction of Eqa   0.022131
Time (sec)                 1.88526
Normalized Time           66.1684
========================  =========

Sineagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.384615
Weighted Fraction of Eqa   0.0923077
Time (sec)                50.8428
Normalized Time           23.9245
========================  ==========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.766667
Weighted Fraction of Eqa    0.0795238
Time (sec)                  3.17016
Normalized Time           528.501
========================  ===========

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.473333
Weighted Fraction of Eqa    0.0500758
Time (sec)                  1.36313
Normalized Time           137.447
========================  ===========

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.427778
Weighted Fraction of Eqa    0.0565079
Time (sec)                 18.7615
Normalized Time           659.094
========================  ===========

Fictitious Play Long
--------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.950014             0.258589
Pure                      0.949728             0.260027
Random                    0.768362             0.169102
Role biased               0.768092             0.166808
Uniform                   0.578544             0.129868
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Fictitious Play                         0.986316      45.7489
Replicator Dynamics                     0.976392     431.777
Multiplicative Weights Stoch            0.936806       2.15751
Regret Matching                         0.924862       3.58979
Multiplicative Weights Dist             0.913064      43.7823
Multiplicative Weights Bandit           0.809015       3.56242
Regret Minimization                     0.638779   13154.4
Scarf 1                                 0.290388    1093.86
Scarf 5                                 0.290388     691.849
Scarf 30                                0.27516      672.628
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.454545
Weighted Fraction of Eqa    0.0636364
Time (sec)                 39.6055
Normalized Time           980.011
========================  ===========

Local effect
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.766667
Weighted Fraction of Eqa      0.0747475
Time (sec)                  382.611
Normalized Time           16301.4
========================  =============

Polyagg small
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.866667
Weighted Fraction of Eqa     0.0862121
Time (sec)                 257.102
Normalized Time           4861.37
========================  ============

Roshambo
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.333333
Weighted Fraction of Eqa      0.055
Time (sec)                  186.158
Normalized Time           22912.8
========================  ============

Shapley easy
""""""""""""

========================  ==============
Metric                             Value
========================  ==============
Fraction of Eqa                0.5
Weighted Fraction of Eqa       0.0454545
Time (sec)                   144.438
Normalized Time           949555
========================  ==============

Hard
""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.0625
Weighted Fraction of Eqa      0.015625
Time (sec)                  187.545
Normalized Time           13258.9
========================  ============

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.6
Weighted Fraction of Eqa   0.0681818
Time (sec)                 0.290723
Normalized Time           50.9614
========================  ==========

Chicken
"""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.616667
Weighted Fraction of Eqa      0.0700794
Time (sec)                  126.227
Normalized Time           30847
========================  =============

Random
""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.460271
Weighted Fraction of Eqa     0.0571623
Time (sec)                 123.897
Normalized Time           2569.2
========================  ============

Polymatrix
""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.391765
Weighted Fraction of Eqa     0.0408757
Time (sec)                  48.7576
Normalized Time           1518
========================  ============

Normagg small
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.530128
Weighted Fraction of Eqa     0.0532707
Time (sec)                 385.825
Normalized Time           2548.34
========================  ============

Shapley normal
""""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa                1
Weighted Fraction of Eqa       0.1
Time (sec)                   137.957
Normalized Time           908372
========================  ==========

Sineagg small
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.335238
Weighted Fraction of Eqa     0.0542381
Time (sec)                 596.186
Normalized Time           3139.22
========================  ============

Shapley hard
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa                1
Weighted Fraction of Eqa       0.111111
Time (sec)                   117.745
Normalized Time           783899
========================  =============

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.203571
Weighted Fraction of Eqa     0.040754
Time (sec)                 152.122
Normalized Time           8438.48
========================  ===========

Normagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.0909091
Time (sec)                60.4186
Normalized Time           40.9112
========================  ==========

Polyagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0.25
Weighted Fraction of Eqa    0.03125
Time (sec)                  4.47837
Normalized Time           466.257
========================  =========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.160812
Weighted Fraction of Eqa     0.022131
Time (sec)                  42.2715
Normalized Time           1483.64
========================  ===========

Sineagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.461538
Weighted Fraction of Eqa    0.117949
Time (sec)                775.405
Normalized Time           364.873
========================  ==========

Mix
"""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.766667
Weighted Fraction of Eqa      0.0795238
Time (sec)                   85.1407
Normalized Time           14193.9
========================  =============

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.473333
Weighted Fraction of Eqa    0.0500758
Time (sec)                  1.40205
Normalized Time           141.37
========================  ===========

Congestion
""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.427778
Weighted Fraction of Eqa      0.0565079
Time (sec)                  678.21
Normalized Time           23825.6
========================  =============

Scarf 1
-------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                   0.974132             0.974132
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Scarf 5                                 0.98936     0.632481
Scarf 30                                0.974132    0.614909
Regret Matching                         0.387655    0.00328175
Replicator Dynamics                     0.336022    0.394726
Multiplicative Weights Stoch            0.295155    0.00197237
Multiplicative Weights Bandit           0.284351    0.00325673
Fictitious Play                         0.262598    0.0418232
Fictitious Play Long                    0.261298    0.00091419
Multiplicative Weights Dist             0.254301    0.0400254
Regret Minimization                     0.252055   12.0256
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.0909091
Weighted Fraction of Eqa  0.030303
Time (sec)                0.0535238
Normalized Time           1.32441
========================  =========

Local effect
""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.766667
Weighted Fraction of Eqa  0.0858586
Time (sec)                0.0386623
Normalized Time           1.64723
========================  =========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.8
Weighted Fraction of Eqa    0.0795455
Time (sec)                 18.5638
Normalized Time           351.011
========================  ===========

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.316667
Weighted Fraction of Eqa  0.0516667
Time (sec)                0.015427
Normalized Time           1.8988
========================  =========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.5
Weighted Fraction of Eqa  0.0454545
Time (sec)                0.000172377
Normalized Time           1.13323
========================  ===========

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.0625
Weighted Fraction of Eqa   0.0208333
Time (sec)                 0.489335
Normalized Time           34.5946
========================  ==========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.6
Weighted Fraction of Eqa  0.151515
Time (sec)                0.00878277
Normalized Time           1.53955
========================  ==========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.616667
Weighted Fraction of Eqa  0.0844444
Time (sec)                0.00419412
Normalized Time           1.02495
========================  ==========

Random
""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.301873
Weighted Fraction of Eqa  0.035978
Time (sec)                0.164147
Normalized Time           3.40386
========================  ========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.341765
Weighted Fraction of Eqa   0.0445276
Time (sec)                 1.92769
Normalized Time           60.0159
========================  ==========

Normagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.430128
Weighted Fraction of Eqa  0.0464064
Time (sec)                0.333739
Normalized Time           2.20432
========================  =========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.1
Time (sec)                0.000175238
Normalized Time           1.15385
========================  ===========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.265238
Weighted Fraction of Eqa   0.0828571
Time (sec)                 4.46379
Normalized Time           23.5041
========================  ==========

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.111111
Time (sec)                0.000174046
Normalized Time           1.15873
========================  ===========

Zero sum
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.203571
Weighted Fraction of Eqa   0.0650794
Time (sec)                 0.325674
Normalized Time           18.0657
========================  ==========

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.0909091
Time (sec)                1.48228
Normalized Time           1.00369
========================  =========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.25
Weighted Fraction of Eqa  0.0833333
Time (sec)                0.00974584
Normalized Time           1.01467
========================  ==========

Covariant
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.107241
Weighted Fraction of Eqa   0.0295616
Time (sec)                 0.335998
Normalized Time           11.7928
========================  ==========

Sineagg large
"""""""""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa            0
Weighted Fraction of Eqa   0
Time (sec)                60.0015
Normalized Time           28.2342
========================  =======

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.766667
Weighted Fraction of Eqa  0.0922222
Time (sec)                0.00621672
Normalized Time           1.0364
========================  ==========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.406667
Weighted Fraction of Eqa   0.0709091
Time (sec)                 0.138762
Normalized Time           13.9916
========================  ==========

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.427778
Weighted Fraction of Eqa    0.0853704
Time (sec)                  2.90812
Normalized Time           102.163
========================  ===========

Multiplicative Weights Bandit
-----------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Random                    0.767292            0.25126
Biased                    0.735855            0.222227
Role biased               0.704566            0.193296
Pure                      0.644773            0.159152
Uniform                   0.326478            0.0795651
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

============================  =================  ============
Method                          Fraction of Eqa    Time Ratio
============================  =================  ============
Regret Matching                        0.885032      1.00768
Multiplicative Weights Stoch           0.84521       0.60563
Replicator Dynamics                    0.781573    121.203
Fictitious Play Long                   0.706261      0.280708
Fictitious Play                        0.704279     12.8421
Multiplicative Weights Dist            0.681678     12.29
Regret Minimization                    0.493663   3692.53
Scarf 5                                0.27124     194.207
Scarf 1                                0.27124     307.056
Scarf 30                               0.256013    188.812
============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.363636
Weighted Fraction of Eqa    0.0454545
Time (sec)                  5.33401
Normalized Time           131.987
========================  ===========

Local effect
""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.866667
Weighted Fraction of Eqa   0.230303
Time (sec)                 2.10839
Normalized Time           89.8291
========================  =========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.733333
Weighted Fraction of Eqa    0.0945455
Time (sec)                  7.67314
Normalized Time           145.086
========================  ===========

Roshambo
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.133333
Weighted Fraction of Eqa     0.0133333
Time (sec)                  44.0678
Normalized Time           5423.99
========================  ============

Shapley easy
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa                1
Weighted Fraction of Eqa       0.545455
Time (sec)                    43.9461
Normalized Time           288908
========================  =============

Hard
""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa              0.375
Weighted Fraction of Eqa     0.375
Time (sec)                  66.2699
Normalized Time           4685.1
========================  =========

Prisoners
"""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.6
Weighted Fraction of Eqa     0.0681818
Time (sec)                   6.07155
Normalized Time           1064.29
========================  ============

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.583333
Weighted Fraction of Eqa    0.147857
Time (sec)                  2.47037
Normalized Time           603.703
========================  ==========

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.41854
Weighted Fraction of Eqa    0.0882121
Time (sec)                 25.6667
Normalized Time           532.241
========================  ===========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa              0.71549
Weighted Fraction of Eqa     0.26668
Time (sec)                  45.9504
Normalized Time           1430.6
========================  ==========

Normagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa             0.68141
Weighted Fraction of Eqa    0.20498
Time (sec)                 19.9221
Normalized Time           131.584
========================  =========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                0
Weighted Fraction of Eqa       0
Time (sec)                    44.2551
Normalized Time           291396
========================  ===========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.361905
Weighted Fraction of Eqa   0.0619048
Time (sec)                14.8681
Normalized Time           78.2878
========================  ==========

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                0
Weighted Fraction of Eqa       0
Time (sec)                    36.8509
Normalized Time           245339
========================  ===========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.316667
Weighted Fraction of Eqa     0.206944
Time (sec)                  71.3499
Normalized Time           3957.91
========================  ===========

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.0909091
Time (sec)                2.33075
Normalized Time           1.57822
========================  =========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa              0.5
Weighted Fraction of Eqa     0.08125
Time (sec)                  10.4592
Normalized Time           1088.94
========================  ==========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.407353
Weighted Fraction of Eqa     0.2739
Time (sec)                  68.9692
Normalized Time           2420.66
========================  ===========

Sineagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.230769
Weighted Fraction of Eqa   0.0448718
Time (sec)                22.102
Normalized Time           10.4003
========================  ==========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.766667
Weighted Fraction of Eqa    0.0795238
Time (sec)                  2.08458
Normalized Time           347.524
========================  ===========

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.573333
Weighted Fraction of Eqa     0.150076
Time (sec)                  30.0279
Normalized Time           3027.76
========================  ===========

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.283333
Weighted Fraction of Eqa    0.062619
Time (sec)                 25.2072
Normalized Time           885.533
========================  ==========

Multiplicative Weights Dist
---------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.847384             0.237331
Role biased               0.830999             0.238315
Random                    0.797182             0.211292
Pure                      0.549807             0.142173
Uniform                   0.508086             0.155283
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Replicator Dynamics                     0.995246     9.86189
Fictitious Play Long                    0.9877       0.0228403
Fictitious Play                         0.987343     1.04492
Multiplicative Weights Stoch            0.951682     0.0492781
Regret Matching                         0.930645     0.0819918
Multiplicative Weights Bandit           0.816177     0.0813667
Regret Minimization                     0.643166   300.449
Scarf 1                                 0.290388    24.9842
Scarf 5                                 0.290388    15.802
Scarf 30                                0.27516     15.363
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.454545
Weighted Fraction of Eqa    0.0636364
Time (sec)                  5.56737
Normalized Time           137.761
========================  ===========

Local effect
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.766667
Weighted Fraction of Eqa    0.0747475
Time (sec)                 14.9307
Normalized Time           636.133
========================  ===========

Polyagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.933333
Weighted Fraction of Eqa    0.152879
Time (sec)                 34.3309
Normalized Time           649.139
========================  ==========

Roshambo
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.4
Weighted Fraction of Eqa    0.0772222
Time (sec)                  2.98553
Normalized Time           367.467
========================  ===========

Shapley easy
""""""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.5
Weighted Fraction of Eqa      0.0454545
Time (sec)                    1.97916
Normalized Time           13011.3
========================  =============

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.25
Weighted Fraction of Eqa    0.203125
Time (sec)                  5.12928
Normalized Time           362.626
========================  ==========

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.6
Weighted Fraction of Eqa    0.0681818
Time (sec)                  2.66754
Normalized Time           467.599
========================  ===========

Chicken
"""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.616667
Weighted Fraction of Eqa    0.0700794
Time (sec)                  2.6405
Normalized Time           645.281
========================  ===========

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.49186
Weighted Fraction of Eqa    0.0799336
Time (sec)                  5.20911
Normalized Time           108.019
========================  ===========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.570588
Weighted Fraction of Eqa    0.178967
Time (sec)                  5.32387
Normalized Time           165.751
========================  ==========

Normagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.571154
Weighted Fraction of Eqa    0.0942963
Time (sec)                 38.9213
Normalized Time           257.072
========================  ===========

Shapley normal
""""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa               1
Weighted Fraction of Eqa      0.1
Time (sec)                    2.0859
Normalized Time           13734.6
========================  ==========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.440952
Weighted Fraction of Eqa    0.159952
Time (sec)                 39.949
Normalized Time           210.351
========================  ==========

Shapley hard
""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               1
Weighted Fraction of Eqa      0.111111
Time (sec)                    2.08389
Normalized Time           13873.7
========================  ============

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.253571
Weighted Fraction of Eqa    0.0949206
Time (sec)                  4.52883
Normalized Time           251.222
========================  ===========

Normagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.0909091
Time (sec)                588.663
Normalized Time           398.601
========================  ===========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa              0.75
Weighted Fraction of Eqa     0.33125
Time (sec)                  41.2665
Normalized Time           4296.39
========================  ==========

Covariant
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.265014
Weighted Fraction of Eqa    0.114568
Time (sec)                  4.82658
Normalized Time           169.402
========================  ==========

Sineagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.769231
Weighted Fraction of Eqa   0.316667
Time (sec)                56.4428
Normalized Time           26.5596
========================  =========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.766667
Weighted Fraction of Eqa    0.0795238
Time (sec)                  2.49913
Normalized Time           416.634
========================  ===========

Rbf
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.613333
Weighted Fraction of Eqa     0.140076
Time (sec)                  10.3394
Normalized Time           1042.54
========================  ===========

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.516667
Weighted Fraction of Eqa    0.119471
Time (sec)                 17.5355
Normalized Time           616.024
========================  ==========

Regret Minimization
-------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.612662            0.225584
Random                    0.570811            0.223032
Pure                      0.542939            0.28933
Role biased               0.505225            0.16979
Uniform                   0.329267            0.0770974
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Replicator Dynamics                     0.951445   0.0328238
Regret Matching                         0.924011   0.000272897
Fictitious Play Long                    0.885018   7.60204e-05
Fictitious Play                         0.883195   0.00347785
Multiplicative Weights Dist             0.860956   0.00332835
Multiplicative Weights Stoch            0.859002   0.000164015
Multiplicative Weights Bandit           0.797547   0.000270817
Scarf 5                                 0.280021   0.0525946
Scarf 1                                 0.280021   0.083156
Scarf 30                                0.264793   0.0511334
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.909091
Weighted Fraction of Eqa  0.518182
Time (sec)                0.0404133
Normalized Time           1
========================  =========

Local effect
""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.766667
Weighted Fraction of Eqa  0.0747475
Time (sec)                0.0234711
Normalized Time           1
========================  =========

Polyagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.866667
Weighted Fraction of Eqa  0.0862121
Time (sec)                0.0528868
Normalized Time           1
========================  =========

Roshambo
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.366667
Weighted Fraction of Eqa  0.143333
Time (sec)                0.00812461
Normalized Time           1
========================  ==========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa            0.5
Weighted Fraction of Eqa   0.0454545
Time (sec)                 0.00493105
Normalized Time           32.4174
========================  ===========

Hard
""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.3125
Weighted Fraction of Eqa  0.3125
Time (sec)                0.0141448
Normalized Time           1
========================  =========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.6
Weighted Fraction of Eqa  0.0681818
Time (sec)                0.00570476
Normalized Time           1
========================  ==========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.616667
Weighted Fraction of Eqa  0.0700794
Time (sec)                0.00568558
Normalized Time           1.38943
========================  ==========

Random
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.711628
Weighted Fraction of Eqa  0.371843
Time (sec)                0.0482239
Normalized Time           1
========================  =========

Polymatrix
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.546863
Weighted Fraction of Eqa  0.152497
Time (sec)                0.0321197
Normalized Time           1
========================  =========

Normagg small
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.721795
Weighted Fraction of Eqa  0.245365
Time (sec)                0.151402
Normalized Time           1
========================  ========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.1
Time (sec)                 0.00519696
Normalized Time           34.2192
========================  ===========

Sineagg small
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.409048
Weighted Fraction of Eqa  0.309714
Time (sec)                0.189915
Normalized Time           1
========================  ========

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.111111
Time (sec)                 0.00591425
Normalized Time           39.3749
========================  ===========

Zero sum
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.153571
Weighted Fraction of Eqa  0.110119
Time (sec)                0.0180272
Normalized Time           1
========================  =========

Normagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.0909091
Time (sec)                22.9034
Normalized Time           15.5086
========================  ==========

Polyagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa            0.5
Weighted Fraction of Eqa   0.08125
Time (sec)                 0.53791
Normalized Time           56.0036
========================  ========

Covariant
"""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.346919
Weighted Fraction of Eqa  0.208237
Time (sec)                0.0284918
Normalized Time           1
========================  =========

Sineagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.153846
Weighted Fraction of Eqa  0.0192308
Time (sec)                2.12514
Normalized Time           1
========================  =========

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.833333
Weighted Fraction of Eqa  0.14619
Time (sec)                0.00599839
Normalized Time           1
========================  ==========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.58
Weighted Fraction of Eqa  0.156742
Time (sec)                0.00991753
Normalized Time           1
========================  ==========

Congestion
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.266667
Weighted Fraction of Eqa  0.120952
Time (sec)                0.0284656
Normalized Time           1
========================  =========

Multiplicative Weights Stoch
----------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.908676             0.226147
Random                    0.899611             0.226593
Role biased               0.897554             0.224031
Pure                      0.664599             0.158811
Uniform                   0.526116             0.120572
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Regret Matching                         0.917777      1.66386
Replicator Dynamics                     0.858239    200.127
Fictitious Play Long                    0.85494       0.463497
Fictitious Play                         0.852829     21.2045
Multiplicative Weights Bandit           0.848466      1.65117
Multiplicative Weights Dist             0.798835     20.293
Regret Minimization                     0.576785   6097.01
Scarf 5                                 0.282456    320.67
Scarf 1                                 0.282456    507.003
Scarf 30                                0.267228    311.761
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.363636
Weighted Fraction of Eqa    0.0454545
Time (sec)                 14.5203
Normalized Time           359.295
========================  ===========

Local effect
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.766667
Weighted Fraction of Eqa    0.0747475
Time (sec)                  2.95129
Normalized Time           125.741
========================  ===========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.666667
Weighted Fraction of Eqa    0.0612121
Time (sec)                  9.94448
Normalized Time           188.033
========================  ===========

Roshambo
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.4
Weighted Fraction of Eqa     0.0772222
Time (sec)                  77.7543
Normalized Time           9570.23
========================  ============

Shapley easy
""""""""""""

========================  ==============
Metric                             Value
========================  ==============
Fraction of Eqa                0.5
Weighted Fraction of Eqa       0.0454545
Time (sec)                    24.7512
Normalized Time           162718
========================  ==============

Hard
""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.0625
Weighted Fraction of Eqa     0.015625
Time (sec)                  96.4727
Normalized Time           6820.35
========================  ===========

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.6
Weighted Fraction of Eqa    0.0681818
Time (sec)                  4.79941
Normalized Time           841.3
========================  ===========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.933333
Weighted Fraction of Eqa    0.238413
Time (sec)                  0.803667
Normalized Time           196.398
========================  ==========

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.470478
Weighted Fraction of Eqa     0.10504
Time (sec)                  70.9756
Normalized Time           1471.79
========================  ===========

Polymatrix
""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.445098
Weighted Fraction of Eqa     0.0613201
Time (sec)                 108.888
Normalized Time           3390.06
========================  ============

Normagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.555769
Weighted Fraction of Eqa    0.0793391
Time (sec)                 23.2277
Normalized Time           153.417
========================  ===========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa                1
Weighted Fraction of Eqa       0.1
Time (sec)                    89.7088
Normalized Time           590684
========================  ===========

Sineagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.315238
Weighted Fraction of Eqa    0.0492381
Time (sec)                 26.366
Normalized Time           138.83
========================  ===========

Shapley hard
""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa                0
Weighted Fraction of Eqa       0
Time (sec)                    87.851
Normalized Time           584879
========================  ==========

Zero sum
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.203571
Weighted Fraction of Eqa      0.040754
Time (sec)                  219.085
Normalized Time           12153
========================  ============

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.0909091
Time (sec)                4.07224
Normalized Time           2.75744
========================  =========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa              0.5
Weighted Fraction of Eqa     0.08125
Time (sec)                  18.0471
Normalized Time           1878.94
========================  ==========

Covariant
"""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.184342
Weighted Fraction of Eqa     0.0338957
Time (sec)                 110.969
Normalized Time           3894.76
========================  ============

Sineagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.307692
Weighted Fraction of Eqa   0.0538462
Time (sec)                34.4364
Normalized Time           16.2044
========================  ==========

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.866667
Weighted Fraction of Eqa    0.179524
Time (sec)                  0.965105
Normalized Time           160.894
========================  ==========

Rbf
"""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.473333
Weighted Fraction of Eqa     0.0500758
Time (sec)                  54.8186
Normalized Time           5527.44
========================  ============

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.572222
Weighted Fraction of Eqa    0.136138
Time (sec)                 16.5029
Normalized Time           579.749
========================  ==========

Scarf 30
--------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                          1                    1
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Scarf 5                                 1           1.02858
Scarf 1                                 1           1.62626
Regret Matching                         0.392975    0.00533696
Replicator Dynamics                     0.346662    0.641925
Multiplicative Weights Stoch            0.305795    0.00320759
Multiplicative Weights Bandit           0.289672    0.00529628
Fictitious Play                         0.273239    0.0680152
Fictitious Play Long                    0.271938    0.00148671
Regret Minimization                     0.262695   19.5567
Multiplicative Weights Dist             0.259621    0.0650915
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.0909091
Weighted Fraction of Eqa  0.030303
Time (sec)                0.0525558
Normalized Time           1.30046
========================  =========

Local effect
""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.766667
Weighted Fraction of Eqa  0.0858586
Time (sec)                0.0385316
Normalized Time           1.64166
========================  =========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.866667
Weighted Fraction of Eqa    0.0862121
Time (sec)                 36.0073
Normalized Time           680.838
========================  ===========

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.316667
Weighted Fraction of Eqa  0.0516667
Time (sec)                0.0152142
Normalized Time           1.8726
========================  =========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.5
Weighted Fraction of Eqa  0.0454545
Time (sec)                0.000152111
Normalized Time           1
========================  ===========

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.0625
Weighted Fraction of Eqa   0.0208333
Time (sec)                 0.489961
Normalized Time           34.6389
========================  ==========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.6
Weighted Fraction of Eqa  0.151515
Time (sec)                0.00862207
Normalized Time           1.51138
========================  ==========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.616667
Weighted Fraction of Eqa  0.0844444
Time (sec)                0.00409203
Normalized Time           1
========================  ==========

Random
""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.301873
Weighted Fraction of Eqa  0.035978
Time (sec)                0.163609
Normalized Time           3.3927
========================  ========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.341765
Weighted Fraction of Eqa   0.0445276
Time (sec)                 1.93136
Normalized Time           60.1302
========================  ==========

Normagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.430128
Weighted Fraction of Eqa  0.0464064
Time (sec)                0.333596
Normalized Time           2.20337
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

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.265238
Weighted Fraction of Eqa   0.0828571
Time (sec)                 4.45589
Normalized Time           23.4625
========================  ==========

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.111111
Time (sec)                0.000150204
Normalized Time           1
========================  ===========

Zero sum
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.203571
Weighted Fraction of Eqa   0.0650794
Time (sec)                 0.325639
Normalized Time           18.0638
========================  ==========

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.0909091
Time (sec)                1.48413
Normalized Time           1.00495
========================  =========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.25
Weighted Fraction of Eqa  0.0833333
Time (sec)                0.00960493
Normalized Time           1
========================  ==========

Covariant
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.107241
Weighted Fraction of Eqa   0.0295616
Time (sec)                 0.337387
Normalized Time           11.8415
========================  ==========

Sineagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.0769231
Weighted Fraction of Eqa    0.0769231
Time (sec)                595.857
Normalized Time           280.385
========================  ===========

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.766667
Weighted Fraction of Eqa  0.0922222
Time (sec)                0.00608792
Normalized Time           1.01493
========================  ==========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.406667
Weighted Fraction of Eqa   0.0709091
Time (sec)                 0.138718
Normalized Time           13.9871
========================  ==========

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.427778
Weighted Fraction of Eqa    0.0853704
Time (sec)                  2.91126
Normalized Time           102.273
========================  ===========

Scarf 5
-------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Uniform                   0.984772             0.984772
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================  =================  ============
Method                           Fraction of Eqa    Time Ratio
=============================  =================  ============
Scarf 1                                 1           1.58107
Scarf 30                                0.984772    0.972218
Regret Matching                         0.392975    0.00518869
Replicator Dynamics                     0.346662    0.624091
Multiplicative Weights Stoch            0.305795    0.00311847
Multiplicative Weights Bandit           0.289672    0.00514914
Fictitious Play                         0.273239    0.0661256
Fictitious Play Long                    0.271938    0.0014454
Regret Minimization                     0.262695   19.0134
Multiplicative Weights Dist             0.259621    0.0632831
=============================  =================  ============

By Game Type
^^^^^^^^^^^^

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.0909091
Weighted Fraction of Eqa  0.030303
Time (sec)                0.0530405
Normalized Time           1.31245
========================  =========

Local effect
""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.766667
Weighted Fraction of Eqa  0.0858586
Time (sec)                0.0384745
Normalized Time           1.63923
========================  =========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.866667
Weighted Fraction of Eqa    0.0862121
Time (sec)                 36.0208
Normalized Time           681.092
========================  ===========

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.316667
Weighted Fraction of Eqa  0.0516667
Time (sec)                0.015255
Normalized Time           1.87763
========================  =========

Shapley easy
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           0.5
Weighted Fraction of Eqa  0.0454545
Time (sec)                0.000152588
Normalized Time           1.00313
========================  ===========

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.0625
Weighted Fraction of Eqa   0.0208333
Time (sec)                 0.5071
Normalized Time           35.8506
========================  ==========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.6
Weighted Fraction of Eqa  0.151515
Time (sec)                0.00864849
Normalized Time           1.51601
========================  ==========

Chicken
"""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.616667
Weighted Fraction of Eqa  0.0844444
Time (sec)                0.00409756
Normalized Time           1.00135
========================  ==========

Random
""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.301873
Weighted Fraction of Eqa  0.035978
Time (sec)                0.168064
Normalized Time           3.48508
========================  ========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.341765
Weighted Fraction of Eqa   0.0445276
Time (sec)                 1.92887
Normalized Time           60.0526
========================  ==========

Normagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.430128
Weighted Fraction of Eqa  0.0464064
Time (sec)                0.333221
Normalized Time           2.2009
========================  =========

Shapley normal
""""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.1
Time (sec)                0.000154972
Normalized Time           1.02041
========================  ===========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.265238
Weighted Fraction of Eqa   0.0828571
Time (sec)                 4.45887
Normalized Time           23.4782
========================  ==========

Shapley hard
""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.111111
Time (sec)                0.000151873
Normalized Time           1.01111
========================  ===========

Zero sum
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.203571
Weighted Fraction of Eqa   0.0650794
Time (sec)                 0.325366
Normalized Time           18.0486
========================  ==========

Normagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.0909091
Time (sec)                1.47682
Normalized Time           1
========================  =========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.25
Weighted Fraction of Eqa  0.0833333
Time (sec)                0.00965047
Normalized Time           1.00474
========================  ==========

Covariant
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.107241
Weighted Fraction of Eqa   0.0295616
Time (sec)                 0.336573
Normalized Time           11.813
========================  ==========

Sineagg large
"""""""""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                300.001
Normalized Time           141.168
========================  =======

Mix
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           0.766667
Weighted Fraction of Eqa  0.0922222
Time (sec)                0.00610781
Normalized Time           1.01824
========================  ==========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.406667
Weighted Fraction of Eqa   0.0709091
Time (sec)                 0.138737
Normalized Time           13.9891
========================  ==========

Congestion
""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.427778
Weighted Fraction of Eqa    0.0853704
Time (sec)                  2.90854
Normalized Time           102.177
========================  ===========

