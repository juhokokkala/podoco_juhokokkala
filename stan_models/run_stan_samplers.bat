@ECHO OFF
REM Windows style batch file for running the Stan samplers
REM This is part of Juho Kokkala's PoDoCo project

REM Copyright (c) 2016 Juho Kokkala
REM This file is licensed under the MIT License.


incoming_poisson sample num_samples=10000 init=0 random seed=2 data^
                 file=../data/incoming.stan.data output^
                 file=output_ic_poisson.csv refresh=1


incoming_poisson_2 sample num_samples=10000 init=0 random seed=2 data^ 
                 file=../data/incoming.stan.data output^
                 file=output_ic_poisson_2.csv refresh=1


incoming_negbin sample num_samples=10000 init=0 random seed=2 data^ 
                 file=../data/incoming.stan.data output^
                 file=output_ic_negbin.csv refresh=1


if_og_negbin sample init=0 random seed=2 data file=../data/3component.stan.data^
             output file=output_ifog_negbin.csv refresh=1


