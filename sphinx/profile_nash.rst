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

======================================================  =================  ===================  ============  =================
Method                                                    Fraction of Eqa    Weighted Fraction    Time (sec)    Normalized Time
======================================================  =================  ===================  ============  =================
Regret Minimization (optimize)                                   0.830632            0.293126       689.321            398.165
EXP3 (regret)                                                    0.771877            0.191869      2826.01            3164.49
Fictitious Play (fictitious)                                     0.670647            0.13122       1040.07            1693.27
Replicator Dynamics (replicator)                                 0.635327            0.127066       228.229             97.3938
EXP3 Pure Deviations EXP3 Pure Deviations (regret_dev)           0.590742            0.128833       186.613           2970.17
EXP3 Single Payoff EXP3 Single Payoff (regret_pay)               0.36308             0.0535562       14.6804            77.5631
Regret Matching (matching)                                       0.362494            0.0688347      220.124           2931.62
======================================================  =================  ===================  ============  =================

EXP3 Pure Deviations EXP3 Pure Deviations (regret_dev)
------------------------------------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.221314            0.0742364
Random                    0.186377            0.0553425
Role biased               0.184569            0.0495586
Pure                      0.117061            0.0342027
Uniform                   0.100895            0.0250351
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

==================================================  =================  ============
Method                                                Fraction of Eqa    Time Ratio
==================================================  =================  ============
EXP3 Single Payoff EXP3 Single Payoff (regret_pay)           0.308758     38.2936
Regret Matching (matching)                                   0.302219      1.01315
Replicator Dynamics (replicator)                             0.252971     30.4965
Fictitious Play (fictitious)                                 0.248094      1.7541
EXP3 (regret)                                                0.243106      0.938595
Regret Minimization (optimize)                               0.197892      7.45965
==================================================  =================  ============

By Game Type
^^^^^^^^^^^^

Roshambo
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.7
Weighted Fraction of Eqa      0.208333
Time (sec)                   52.5124
Normalized Time           14912.8
========================  ============

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.379744
Weighted Fraction of Eqa    0.0542491
Time (sec)                157.671
Normalized Time            66.0547
========================  ===========

Chicken
"""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.215
Time (sec)                 23.9455
Normalized Time           133.039
========================  ========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.8775
Weighted Fraction of Eqa    0.211756
Time (sec)                316.09
Normalized Time            69.3032
========================  ==========

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.142857
Time (sec)                  20.0756
Normalized Time           8201.43
========================  ===========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.559967
Weighted Fraction of Eqa    0.115562
Time (sec)                786.223
Normalized Time            19.8095
========================  ==========

Shapley
"""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                229.813
Normalized Time           633.119
========================  =======

Sineagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.500117
Weighted Fraction of Eqa   0.11337
Time (sec)                84.0581
Normalized Time            3.76734
========================  =========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.391667
Weighted Fraction of Eqa     0.126025
Time (sec)                 219.513
Normalized Time           8687.26
========================  ===========

Covariant
"""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.771429
Weighted Fraction of Eqa      0.211104
Time (sec)                  488.459
Normalized Time           12867.3
========================  ============

Normagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.9
Weighted Fraction of Eqa    0.128571
Time (sec)                125.682
Normalized Time           102.257
========================  ==========

Local effect
""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.4
Weighted Fraction of Eqa   0.0628571
Time (sec)                49.1204
Normalized Time           11.574
========================  ==========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.44885
Weighted Fraction of Eqa    0.180371
Time (sec)                182.665
Normalized Time            18.8571
========================  ==========

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.444444
Weighted Fraction of Eqa    0.0634921
Time (sec)                177.168
Normalized Time           977.24
========================  ===========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.173333
Time (sec)                  23.9438
Normalized Time           1798.68
========================  ===========

Sineagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.134557
Weighted Fraction of Eqa    0.0259336
Time (sec)                174.647
Normalized Time             5.55297
========================  ===========

Normagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.713333
Weighted Fraction of Eqa   0.167619
Time (sec)                76.1766
Normalized Time           10.4776
========================  =========

Hard
""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                458.338
Normalized Time            17.5478
========================  ========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.929902
Weighted Fraction of Eqa     0.201467
Time (sec)                 151.159
Normalized Time           6400.88
========================  ===========

Congestion
""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                171.051
Normalized Time            15.5213
========================  ========

Regret Matching (matching)
--------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Pure                     0.130001            0.0487435
Biased                   0.118625            0.0374456
Role biased              0.0934353           0.0260987
Random                   0.0827933           0.0230695
Uniform                  0.0199017           0.00437663
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

======================================================  =================  ============
Method                                                    Fraction of Eqa    Time Ratio
======================================================  =================  ============
EXP3 Single Payoff EXP3 Single Payoff (regret_pay)               0.302451     37.7966
Replicator Dynamics (replicator)                                 0.215893     30.1007
EXP3 Pure Deviations EXP3 Pure Deviations (regret_dev)           0.19457       0.987022
Fictitious Play (fictitious)                                     0.161004      1.73134
EXP3 (regret)                                                    0.151669      0.926414
Regret Minimization (optimize)                                   0.100262      7.36284
======================================================  =================  ============

By Game Type
^^^^^^^^^^^^

Roshambo
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa               0
Weighted Fraction of Eqa      0
Time (sec)                   38.1763
Normalized Time           10841.5
========================  ==========

Random
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.379744
Weighted Fraction of Eqa    0.0542491
Time (sec)                225.595
Normalized Time            94.5109
========================  ===========

Chicken
"""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                 22.5057
Normalized Time           125.039
========================  ========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.7
Weighted Fraction of Eqa    0.136339
Time (sec)                312.338
Normalized Time            68.4806
========================  ==========

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.142857
Time (sec)                  13.2016
Normalized Time           5393.2
========================  ===========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.473802
Weighted Fraction of Eqa    0.067686
Time (sec)                177.043
Normalized Time             4.46075
========================  ==========

Shapley
"""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                198.358
Normalized Time           546.461
========================  =======

Sineagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.300117
Weighted Fraction of Eqa    0.0561476
Time (sec)                142.411
Normalized Time             6.38263
========================  ===========

Zero sum
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.255556
Weighted Fraction of Eqa     0.0690807
Time (sec)                 208.656
Normalized Time           8257.58
========================  ============

Covariant
"""""""""

========================  =============
Metric                            Value
========================  =============
Fraction of Eqa               0.392778
Weighted Fraction of Eqa      0.0576984
Time (sec)                  394.095
Normalized Time           10381.5
========================  =============

Normagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.9
Weighted Fraction of Eqa    0.128571
Time (sec)                416.895
Normalized Time           339.193
========================  ==========

Local effect
""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.3
Weighted Fraction of Eqa   0.0428571
Time (sec)                78.1722
Normalized Time           18.4192
========================  ==========

Polyagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.352954
Weighted Fraction of Eqa    0.211695
Time (sec)                434.959
Normalized Time            44.9022
========================  ==========

Gambit
""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.444444
Weighted Fraction of Eqa    0.0634921
Time (sec)                172.317
Normalized Time           950.483
========================  ===========

Mix
"""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa              0
Weighted Fraction of Eqa     0
Time (sec)                  27.397
Normalized Time           2058.09
========================  ========

Sineagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.12482
Weighted Fraction of Eqa    0.0229365
Time (sec)                423.485
Normalized Time            13.4649
========================  ===========

Normagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.473333
Weighted Fraction of Eqa    0.067619
Time (sec)                106.507
Normalized Time            14.6493
========================  ==========

Hard
""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                801.685
Normalized Time            30.6931
========================  ========

Polyagg small
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.799837
Weighted Fraction of Eqa      0.168705
Time (sec)                  354.685
Normalized Time           15019.2
========================  ============

Congestion
""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             0.1
Weighted Fraction of Eqa    0.02
Time (sec)                334.368
Normalized Time            30.3407
========================  ========

EXP3 (regret)
-------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                   0.237054             0.0752998
Role biased              0.205905             0.0576887
Random                   0.203299             0.0598957
Uniform                  0.172423             0.0444531
Pure                     0.0984478            0.0263452
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

======================================================  =================  ============
Method                                                    Fraction of Eqa    Time Ratio
======================================================  =================  ============
EXP3 Single Payoff EXP3 Single Payoff (regret_pay)               0.308203      40.7989
Regret Matching (matching)                                       0.303702       1.07943
Replicator Dynamics (replicator)                                 0.302257      32.4917
Fictitious Play (fictitious)                                     0.294828       1.86886
EXP3 Pure Deviations EXP3 Pure Deviations (regret_dev)           0.288423       1.06542
Regret Minimization (optimize)                                   0.216205       7.94768
======================================================  =================  ============

By Game Type
^^^^^^^^^^^^

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa              0.5
Weighted Fraction of Eqa     0.125
Time (sec)                  21.7736
Normalized Time           6183.4
========================  =========

Random
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.580421
Weighted Fraction of Eqa   0.137322
Time (sec)                60.1199
Normalized Time           25.1866
========================  =========

Chicken
"""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa            1
Weighted Fraction of Eqa   0.215
Time (sec)                15.8193
Normalized Time           87.8903
========================  =======

Polymatrix
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.79
Weighted Fraction of Eqa   0.174256
Time (sec)                61.8156
Normalized Time           13.5532
========================  =========

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.142857
Time (sec)                  15.9195
Normalized Time           6503.53
========================  ===========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.724416
Weighted Fraction of Eqa    0.249763
Time (sec)                248.957
Normalized Time             6.27268
========================  ==========

Shapley
"""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa             1
Weighted Fraction of Eqa    0.25
Time (sec)                 40.676
Normalized Time           112.059
========================  =======

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.746183
Weighted Fraction of Eqa    0.208539
Time (sec)                330.617
Normalized Time            14.8177
========================  ==========

Zero sum
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.316667
Weighted Fraction of Eqa     0.0940807
Time (sec)                  37.5889
Normalized Time           1487.59
========================  ============

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.765873
Weighted Fraction of Eqa     0.199993
Time (sec)                  81.169
Normalized Time           2138.2
========================  ===========

Normagg large
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               1
Weighted Fraction of Eqa      0.153571
Time (sec)                12352.5
Normalized Time           10050.2
========================  ============

Local effect
""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.212857
Time (sec)                95.6399
Normalized Time           22.5351
========================  =========

Polyagg large
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.472483
Weighted Fraction of Eqa      0.201004
Time (sec)                29473.6
Normalized Time            3042.66
========================  ============

Gambit
""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.666667
Weighted Fraction of Eqa    0.21164
Time (sec)                 50.0129
Normalized Time           275.866
========================  ==========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.173333
Time (sec)                  15.0002
Normalized Time           1126.83
========================  ===========

Sineagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.401717
Weighted Fraction of Eqa     0.130322
Time (sec)                7436.15
Normalized Time            236.435
========================  ===========

Normagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.818333
Weighted Fraction of Eqa    0.255952
Time (sec)                352.945
Normalized Time            48.5451
========================  ==========

Hard
""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.910256
Time (sec)                95.5071
Normalized Time            3.65656
========================  =========

Polyagg small
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa               0.941013
Weighted Fraction of Eqa      0.212578
Time (sec)                  624.674
Normalized Time           26452
========================  ============

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.824396
Weighted Fraction of Eqa    0.243403
Time (sec)                153.744
Normalized Time            13.9509
========================  ==========

Regret Minimization (optimize)
------------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Pure                      0.261808            0.0832775
Biased                    0.246648            0.0657031
Random                    0.234722            0.0636001
Role biased               0.219243            0.0530473
Uniform                   0.19222             0.0418865
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

======================================================  =================  ============
Method                                                    Fraction of Eqa    Time Ratio
======================================================  =================  ============
EXP3 Single Payoff EXP3 Single Payoff (regret_pay)               0.307465      5.13343
Regret Matching (matching)                                       0.297748      0.135817
Replicator Dynamics (replicator)                                 0.288925      4.0882
EXP3 Pure Deviations EXP3 Pure Deviations (regret_dev)           0.285834      0.134055
Fictitious Play (fictitious)                                     0.28571       0.235146
EXP3 (regret)                                                    0.260818      0.125823
======================================================  =================  ============

By Game Type
^^^^^^^^^^^^

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.508333
Time (sec)                 0.190019
Normalized Time           53.9629
========================  =========

Random
""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.904304
Weighted Fraction of Eqa  0.501355
Time (sec)                5.01977
Normalized Time           2.10298
========================  ========

Chicken
"""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.95
Weighted Fraction of Eqa  0.19
Time (sec)                0.179989
Normalized Time           1
========================  ========

Polymatrix
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.7725
Weighted Fraction of Eqa   0.193214
Time (sec)                10.7645
Normalized Time            2.36013
========================  =========

Prisoners
"""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.142857
Time (sec)                 0.232961
Normalized Time           95.1706
========================  =========

Rbf
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.652258
Weighted Fraction of Eqa   0.246142
Time (sec)                39.6891
Normalized Time            1
========================  =========

Shapley
"""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.25
Time (sec)                 6.28381
Normalized Time           17.3114
========================  ========

Sineagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.756702
Weighted Fraction of Eqa   0.345019
Time (sec)                34.7052
Normalized Time            1.55543
========================  =========

Zero sum
""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.761111
Weighted Fraction of Eqa    0.541766
Time (sec)                  6.21261
Normalized Time           245.865
========================  ==========

Covariant
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.716905
Weighted Fraction of Eqa    0.286845
Time (sec)                 16.8464
Normalized Time           443.778
========================  ==========

Normagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.153571
Time (sec)                2890.43
Normalized Time           2351.71
========================  ===========

Local effect
""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.212857
Time (sec)                5.0116
Normalized Time           1.18085
========================  ========

Polyagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.432073
Weighted Fraction of Eqa     0.321716
Time (sec)                8261.91
Normalized Time            852.903
========================  ===========

Gambit
""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.777778
Weighted Fraction of Eqa  0.396825
Time (sec)                1.48937
Normalized Time           8.21522
========================  ========

Mix
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.173333
Time (sec)                0.0902414
Normalized Time           6.77904
========================  =========

Sineagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.743211
Weighted Fraction of Eqa     0.541719
Time (sec)                1150.34
Normalized Time             36.5756
========================  ===========

Normagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.735
Weighted Fraction of Eqa   0.249286
Time (sec)                29.8929
Normalized Time            4.11156
========================  =========

Hard
""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.153846
Weighted Fraction of Eqa   0.0641026
Time (sec)                26.1194
Normalized Time            1
========================  ==========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.747549
Weighted Fraction of Eqa     0.123063
Time (sec)                  73.8524
Normalized Time           3127.3
========================  ===========

Congestion
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.852721
Weighted Fraction of Eqa   0.307719
Time (sec)                11.2312
Normalized Time            1.01912
========================  =========

Replicator Dynamics (replicator)
--------------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.205774            0.0562447
Role biased               0.187519            0.0466122
Random                    0.178678            0.0442627
Uniform                   0.161787            0.0399481
Pure                      0.118913            0.0281905
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

======================================================  =================  ============
Method                                                    Fraction of Eqa    Time Ratio
======================================================  =================  ============
EXP3 Single Payoff EXP3 Single Payoff (regret_pay)               0.308017     1.25567
Regret Matching (matching)                                       0.297052     0.0332218
Fictitious Play (fictitious)                                     0.243612     0.0575182
EXP3 (regret)                                                    0.229695     0.0307771
EXP3 Pure Deviations EXP3 Pure Deviations (regret_dev)           0.226778     0.0327906
Regret Minimization (optimize)                                   0.165889     0.244607
======================================================  =================  ============

By Game Type
^^^^^^^^^^^^

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0
Weighted Fraction of Eqa   0
Time (sec)                 0.240434
Normalized Time           68.28
========================  =========

Random
""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.540037
Weighted Fraction of Eqa  0.102066
Time (sec)                2.38698
Normalized Time           1
========================  ========

Chicken
"""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.95
Weighted Fraction of Eqa  0.19
Time (sec)                0.695226
Normalized Time           3.8626
========================  ========

Polymatrix
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.5825
Weighted Fraction of Eqa   0.0832143
Time (sec)                20.3436
Normalized Time            4.46036
========================  ==========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.142857
Time (sec)                 0.0268353
Normalized Time           10.9629
========================  ==========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.575569
Weighted Fraction of Eqa    0.134886
Time (sec)                149.158
Normalized Time             3.75815
========================  ==========

Shapley
"""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.25
Time (sec)                0.362986
Normalized Time           1
========================  ========

Sineagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.599592
Weighted Fraction of Eqa   0.131392
Time (sec)                47.4186
Normalized Time            2.12522
========================  =========

Zero sum
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.161111
Weighted Fraction of Eqa    0.0230159
Time (sec)                  2.54665
Normalized Time           100.784
========================  ===========

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.384444
Weighted Fraction of Eqa    0.0549206
Time (sec)                 20.8224
Normalized Time           548.515
========================  ===========

Normagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.153571
Time (sec)                352.532
Normalized Time           286.826
========================  ==========

Local effect
""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.212857
Time (sec)                4.24405
Normalized Time           1
========================  ========

Polyagg large
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.15306
Weighted Fraction of Eqa     0.0355103
Time (sec)                1642.41
Normalized Time            169.552
========================  ============

Gambit
""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.555556
Weighted Fraction of Eqa  0.100529
Time (sec)                0.181294
Normalized Time           1
========================  ========

Mix
"""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.173333
Time (sec)                0.0133118
Normalized Time           1
========================  =========

Sineagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.440899
Weighted Fraction of Eqa     0.16867
Time (sec)                1843.54
Normalized Time             58.6162
========================  ===========

Normagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.598333
Weighted Fraction of Eqa   0.0959524
Time (sec)                30.496
Normalized Time            4.19451
========================  ==========

Hard
""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             0
Weighted Fraction of Eqa    0
Time (sec)                135.683
Normalized Time             5.1947
========================  ========

Polyagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.686438
Weighted Fraction of Eqa    0.0980626
Time (sec)                 11.926
Normalized Time           505.011
========================  ===========

Congestion
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.835411
Weighted Fraction of Eqa   0.252244
Time (sec)                11.0204
Normalized Time            1
========================  =========

Fictitious Play (fictitious)
----------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Pure                      0.265629            0.0676815
Biased                    0.255147            0.0576913
Role biased               0.237706            0.0513369
Random                    0.230769            0.0490614
Uniform                   0.217504            0.0455747
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

======================================================  =================  ============
Method                                                    Fraction of Eqa    Time Ratio
======================================================  =================  ============
EXP3 Single Payoff EXP3 Single Payoff (regret_pay)               0.30722      21.8309
Regret Matching (matching)                                       0.298005      0.577588
Replicator Dynamics (replicator)                                 0.297168     17.3858
EXP3 Pure Deviations EXP3 Pure Deviations (regret_dev)           0.277394      0.570092
EXP3 (regret)                                                    0.269339      0.535085
Regret Minimization (optimize)                                   0.217604      4.25269
======================================================  =================  ============

By Game Type
^^^^^^^^^^^^

Roshambo
""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.6
Weighted Fraction of Eqa     0.158333
Time (sec)                  32.8073
Normalized Time           9316.81
========================  ===========

Random
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.528926
Weighted Fraction of Eqa   0.09651
Time (sec)                59.48
Normalized Time           24.9185
========================  =========

Chicken
"""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             0.95
Weighted Fraction of Eqa    0.19
Time (sec)                 22.8555
Normalized Time           126.982
========================  ========

Polymatrix
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.695
Weighted Fraction of Eqa   0.118006
Time (sec)                51.6523
Normalized Time           11.3249
========================  =========

Prisoners
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.142857
Time (sec)                   2.57305
Normalized Time           1051.16
========================  ===========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.552169
Weighted Fraction of Eqa    0.11272
Time (sec)                181.131
Normalized Time             4.56376
========================  ==========

Shapley
"""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa             1
Weighted Fraction of Eqa    0.25
Time (sec)                 64.6079
Normalized Time           177.99
========================  ========

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.582168
Weighted Fraction of Eqa    0.118135
Time (sec)                285.354
Normalized Time            12.789
========================  ==========

Zero sum
""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.161111
Weighted Fraction of Eqa     0.0230159
Time (sec)                  40.2584
Normalized Time           1593.23
========================  ============

Covariant
"""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.617857
Weighted Fraction of Eqa     0.134517
Time (sec)                  78.7941
Normalized Time           2075.64
========================  ===========

Normagg large
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.153571
Time (sec)                4563.37
Normalized Time           3712.84
========================  ===========

Local effect
""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            1
Weighted Fraction of Eqa   0.212857
Time (sec)                99.9433
Normalized Time           23.549
========================  =========

Polyagg large
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.141154
Weighted Fraction of Eqa     0.0269928
Time (sec)                8348.87
Normalized Time            861.88
========================  ============

Gambit
""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa            0.555556
Weighted Fraction of Eqa   0.100529
Time (sec)                 9.68219
Normalized Time           53.406
========================  =========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              1
Weighted Fraction of Eqa     0.173333
Time (sec)                  22.6162
Normalized Time           1698.95
========================  ===========

Sineagg large
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.319848
Weighted Fraction of Eqa     0.0940957
Time (sec)                4421.57
Normalized Time            140.586
========================  ============

Normagg small
"""""""""""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.598333
Weighted Fraction of Eqa    0.0959524
Time (sec)                192.374
Normalized Time            26.4597
========================  ===========

Hard
""""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa             0.0769231
Weighted Fraction of Eqa    0.025641
Time (sec)                157.001
Normalized Time             6.01091
========================  ===========

Polyagg small
"""""""""""""

========================  ============
Metric                           Value
========================  ============
Fraction of Eqa              0.686438
Weighted Fraction of Eqa     0.0980626
Time (sec)                 234.569
Normalized Time           9932.88
========================  ============

Congestion
""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa             0.709533
Weighted Fraction of Eqa    0.176634
Time (sec)                209.708
Normalized Time            19.029
========================  ==========

EXP3 Single Payoff EXP3 Single Payoff (regret_pay)
--------------------------------------------------

Initial Profile Rates
^^^^^^^^^^^^^^^^^^^^^

===============  =================  ===================
Starting Type      Fraction of Eqa    Weighted Fraction
===============  =================  ===================
Biased                    0.139958            0.0323689
Pure                      0.136726            0.0312367
Role biased               0.135331            0.0311351
Random                    0.129476            0.0288452
Uniform                   0.103206            0.0211024
===============  =================  ===================

Compared to Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

======================================================  =================  ============
Method                                                    Fraction of Eqa    Time Ratio
======================================================  =================  ============
Regret Matching (matching)                                       0.295308     0.0264574
Replicator Dynamics (replicator)                                 0.219513     0.796386
EXP3 Pure Deviations EXP3 Pure Deviations (regret_dev)           0.195097     0.026114
Fictitious Play (fictitious)                                     0.162156     0.0458067
EXP3 (regret)                                                    0.150678     0.0245105
Regret Minimization (optimize)                                   0.103683     0.194801
======================================================  =================  ============

By Game Type
^^^^^^^^^^^^

Roshambo
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0
Weighted Fraction of Eqa  0
Time (sec)                0.0035213
Normalized Time           1
========================  =========

Random
""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.379744
Weighted Fraction of Eqa   0.0542491
Time (sec)                25.5704
Normalized Time           10.7124
========================  ==========

Chicken
"""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa            0
Weighted Fraction of Eqa   0
Time (sec)                14.3555
Normalized Time           79.7574
========================  =======

Polymatrix
""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.5825
Weighted Fraction of Eqa  0.0832143
Time (sec)                4.56097
Normalized Time           1
========================  =========

Prisoners
"""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa           1
Weighted Fraction of Eqa  0.142857
Time (sec)                0.00244782
Normalized Time           1
========================  ==========

Rbf
"""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.484913
Weighted Fraction of Eqa   0.0732415
Time (sec)                65.2239
Normalized Time            1.64337
========================  ==========

Shapley
"""""""

========================  =======
Metric                      Value
========================  =======
Fraction of Eqa            0
Weighted Fraction of Eqa   0
Time (sec)                11.9216
Normalized Time           32.8432
========================  =======

Sineagg small
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.191783
Weighted Fraction of Eqa   0.0273976
Time (sec)                22.3123
Normalized Time            1
========================  ==========

Zero sum
""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.161111
Weighted Fraction of Eqa  0.0230159
Time (sec)                0.0252684
Normalized Time           1
========================  =========

Covariant
"""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.384444
Weighted Fraction of Eqa  0.0549206
Time (sec)                0.0379613
Normalized Time           1
========================  =========

Normagg large
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.9
Weighted Fraction of Eqa  0.128571
Time (sec)                1.22908
Normalized Time           1
========================  ========

Local effect
""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.3
Weighted Fraction of Eqa   0.0428571
Time (sec)                14.1562
Normalized Time            3.33554
========================  ==========

Polyagg large
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.105093
Weighted Fraction of Eqa  0.0227111
Time (sec)                9.68681
Normalized Time           1
========================  =========

Gambit
""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.444444
Weighted Fraction of Eqa   0.0634921
Time (sec)                 9.04562
Normalized Time           49.8947
========================  ==========

Mix
"""

========================  ===========
Metric                          Value
========================  ===========
Fraction of Eqa              0.8
Weighted Fraction of Eqa     0.133333
Time (sec)                  16.8276
Normalized Time           1264.11
========================  ===========

Sineagg large
"""""""""""""

========================  ==========
Metric                         Value
========================  ==========
Fraction of Eqa            0.114259
Weighted Fraction of Eqa   0.0163227
Time (sec)                31.4511
Normalized Time            1
========================  ==========

Normagg small
"""""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa           0.473333
Weighted Fraction of Eqa  0.067619
Time (sec)                7.27046
Normalized Time           1
========================  ========

Hard
""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa            0
Weighted Fraction of Eqa   0
Time (sec)                98.4246
Normalized Time            3.76826
========================  ========

Polyagg small
"""""""""""""

========================  =========
Metric                        Value
========================  =========
Fraction of Eqa           0.686438
Weighted Fraction of Eqa  0.0980626
Time (sec)                0.0236154
Normalized Time           1
========================  =========

Congestion
""""""""""

========================  ========
Metric                       Value
========================  ========
Fraction of Eqa            0
Weighted Fraction of Eqa   0
Time (sec)                31.778
Normalized Time            2.88355
========================  ========

