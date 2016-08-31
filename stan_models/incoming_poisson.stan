/*
 * Poisson Model 1 for incoming traffic
 * This is part of Juho Kokkala's PoDoCo project.
 * ----------------------------------------------------------------------------
 * Copyright Juho Kokkala
 * Year: 2016
 * License: MIT
 */


data {
    int<lower=0> N;                 //Number of intervals in a day
    int<lower=0> D;                 //Number of days
    real dt;                        //Length of interval
    int<lower=0> y_ic[D,N];         //Number of arrivals in each interval
}


parameters {
    real mu;                        //The long-term mean of the baseline
    real<lower=0> invlscale_base;   //Inverse lengthscale of the baseline
    real<lower=0> s_base;           //Standard deviation of the baseline
    real<lower=0> invlscale_local;  //Inv. lengthscale of the local variation
    real<lower=0> s_local;          //Standard deviation of the local variation
    vector[N] x_base_shock;         //Iid N(0,1), transformed into the baseline
    vector[N] x_shock[D];           //Iid N(0,1), transformed into intensities
}


transformed parameters {
    /* Transforming the standard normals into the baseline log-intensity and
     * the log-intensities.
     * For the baseline Matérn 3/2, this is following the approach documented
     * in 
     *  http://www.juhokokkala.fi/blog/posts/linear-time-evaluation-of-matern-
     *  32-gp-density-in-stan-using-the-sde-representation/
     * while the local variation is just Ornstein--Uhlenbeck, so, AR(1), so
     * that part is straightforward.
     */

    vector[N] x_base;               //The baseline log-intensity
    vector[N] x[D];                 //The actual log-intensities

    //Temporary variables for computing the transformations
    real lda;
    real m_basederiv;
    real P_basederiv;
    real m_base;
    real P_base;
    matrix[2,2] A;
    vector[2] Q;
    real covQ;
    real cov;
    real A_local;
    real sqrtQ_local;

    //Constructing the Matérn 3/2 baseline process
    lda = sqrt(3) * invlscale_base;
    A[1,1] = (1 + dt * lda) * exp(-dt * lda);
    A[1,2] = dt * exp(-dt * lda);
    A[2,1] = -dt * lda * lda * exp(-dt * lda);
    A[2,2] = (1 - dt * lda) * exp(-dt * lda);
    Q[1] = s_base * s_base * (1 - exp(-2 * dt * lda) * ((1 + dt * lda) * 
            (1 + dt * lda) + dt * lda * dt * lda));
    Q[2] = s_base * s_base * (lda * lda - exp(-2 * dt * lda) * (lda * lda * 
            (1 - dt * lda) * (1 - dt * lda) + dt * dt * lda * lda * lda *
            lda));
    covQ = 2 * s_base * s_base * dt * dt * lda * lda * lda * exp(-2 * dt * 
            lda);
     
    x_base[1] = mu + s_base * x_base_shock[1];
    m_basederiv = 0; 
    P_basederiv = pow(s_base * lda,2);

    for (n in 2:N) {
        m_base = mu + A[1,1] * (x_base[n-1] - mu) + A[1,2] * m_basederiv;
        P_base = Q[1] + A[1,2] * A[1,2] * P_basederiv;
        x_base[n] = m_base + sqrt(P_base) * x_base_shock[n];
        cov = covQ + A[1,2] * A[2,2] * P_basederiv;
        m_basederiv = A[2,1] * (x_base[n-1] - mu) + A[2,2] * m_basederiv  +
                       (cov / P_base) * (x_base[n] - m_base);
        P_basederiv = A[2,2] * A[2,2] * P_basederiv + Q[2] - 
                       cov * cov / P_base;
    }

    //Constructing the actual log-intensities (AR(1) on top of the baseline)
    A_local = exp(-dt * invlscale_local);
    sqrtQ_local = s_local * sqrt(1 - exp(-2 * dt * invlscale_local));

    for (d in 1:D) {
        x[d][1] = x_base[1] + s_local * x_shock[d][1];
        for (i in 2:N) {
            x[d][i] = x_base[i] + A_local * (x[d][i-1] - x_base[i-1]) +
                       sqrtQ_local * x_shock[d][i];
        }
    }
}


model {
    //MODEL SPECIFICATION
    //Parameter priors
    invlscale_local ~ exponential(1); 
    s_local         ~ student_t(2, 0, 1); 
    invlscale_base  ~ exponential(0.25); 
    s_base          ~ student_t(2, 0, 1);
    //mu            ~ U(-inf, inf)

    //Intensity processes (handled by the transformation)
    //x_base        ~ GP(mu, Matérn-3/2-covariance(params: *_base))
    //x_base[d]     ~ GP(x_base, O--U-covariance(params: *_local))

    //Observation model
    for (d in 1:D) {
       y_ic[d]      ~ poisson_log(x[d]);
    }

    //STAN TRICKERY
    x_base_shock ~ normal(0, 1);
    for (d in 1:D) {
 	x_shock[d] ~ normal(0, 1);
    }      
}
