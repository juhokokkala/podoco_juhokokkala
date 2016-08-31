/*
 * Model for interfloor & outgoing traffic
 * This is part of Juho Kokkala's PoDoCo project.
 * ----------------------------------------------------------------------------
 * Copyright Juho Kokkala
 * Year: 2016
 * License: MIT
 */


data {
    int<lower=0> N;                  //Number of intervals in a day
    int<lower=0> D;                  //Number of days
    real dt;                         //Length of interval
    int<lower=0> y_ic[D,N];          //Number of arrivals in each interval
    int<lower=0> y_if[D,N];          //Number of interfloor trips - -
    int<lower=0> y_og[D,N];          //Number of outgoing trips - -
}


transformed data {
    int net_arrivals[D,N];           //Net new people into building before y_[d,n]
    int<lower=0> N_init[D];          //(Minimum consistent) initial population
    int<lower=0> trips[D,N,3];       //Number of if, og, and no trips per interval

    for (d in 1:D) {
        net_arrivals[d,1] = 0;
        N_init[d] = y_if[d,1] + y_og[d,1];
        for (k in 2:N) {
            net_arrivals[d,k] = net_arrivals[d,k-1] + y_ic[d,k-1] - y_og[d,k-1];
            N_init[d] = max(N_init[d],y_if[d,k]+y_og[d,k]-net_arrivals[d,k]);
        }
    }

    for (d in 1:D) {
        for (k in 1:N) {
            trips[d,k,1] = y_if[d,k];
            trips[d,k,2] = y_og[d,k];
            trips[d,k,3] = N_init[d] + net_arrivals[d,k] - y_if[d,k] - y_og[d,k];
        }
    }
}


parameters {
    real mu_if;                         //Long-term mean of the baseline
    real<lower=0> lscale_base_if;       //Lengthscale of the baseline
    real<lower=0> s_base_if;            //Standard deviation of the baseline
    real<lower=0> lscale_local_if;      //Lengthscale of the local variation
    real<lower=0> s_local_if;           //St. dev. of the local variation
    vector[N] x_base_shock_if;          //Iid N(0,1) transformed to the baseline
    vector[N] x_shock_if[D];            //Iid N(0,1) transformed to intensities
    real<lower=0, upper=1> invomega_if; //Inverse of dispersion parameter
    vector<lower=0>[N] gamma_if[D];     //Gamma-distributed intensity multiplier

    real mu_og;                         //Long-term mean of the baseline
    real<lower=0> lscale_base_og;       //Lengthscale of the baseline
    real<lower=0> s_base_og;            //Standard deviation of the baseline
    real<lower=0> lscale_local_og;      //Lengthscale of the local variation
    real<lower=0> s_local_og;           //St. dev. of the local variation
    vector[N] x_base_shock_og;          //Iid N(0,1) transformed to the baseline
    vector[N] x_shock_og[D];            //Iid N(0,1) transformed to intensities
    real<lower=0, upper=1> invomega_og; //Inverse of dispersion parameter 
    vector<lower=0>[N] gamma_og[D];     //Gamma-distributed intensity multiplier

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

    vector[N] x_base_if;                //The baseline log-intensity
    vector[N] x_if[D];                  //The actual log-intensities
    vector[N] x_base_og;                //The baseline log-intensity
    vector[N] x_og[D];                  //The actual log-intensities
    vector[3] p_trip[D,N];              //Probabs of if, og, no trip

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

    //Constructing x_base_if
    lda = sqrt(3) / lscale_base_if;
    A[1,1] = (1 + dt * lda) * exp(-dt * lda);
    A[1,2] = dt * exp(-dt * lda);
    A[2,1] = -dt * lda * lda * exp(-dt * lda);
    A[2,2] = (1 - dt * lda) * exp(-dt * lda);
    Q[1] = s_base_if * s_base_if * (1 - exp(-2 * dt * lda) * ((1 + dt * lda) * 
            (1 + dt * lda) + dt * lda * dt * lda));
    Q[2] = s_base_if * s_base_if * (lda * lda - exp(-2 * dt * lda) * (lda * lda * 
            (1 - dt * lda) * (1 - dt * lda) + dt * dt * lda * lda * lda *
            lda));
    covQ = 2 * s_base_if * s_base_if * dt * dt * lda * lda * lda * exp(-2 * dt * lda);
     
    x_base_if[1] = mu_if + s_base_if * x_base_shock_if[1];
    m_basederiv = 0; 
    P_basederiv = pow(s_base_if * lda,2);

    for (n in 2:N) {
        m_base = mu_if + A[1,1] * (x_base_if[n-1] - mu_if) + A[1,2] * m_basederiv;
        P_base = Q[1] + A[1,2] * A[1,2] * P_basederiv;
        x_base_if[n] = m_base + sqrt(P_base) * x_base_shock_if[n];
        cov = covQ + A[1,2] * A[2,2] * P_basederiv;
        m_basederiv = A[2,1] * (x_base_if[n-1] - mu_if) + A[2,2] * m_basederiv  +
                       (cov / P_base) * (x_base_if[n] - m_base);
        P_basederiv = A[2,2] * A[2,2] * P_basederiv + Q[2] - 
                       cov * cov / P_base;
    }

    //Constructing x_base_og
    lda = sqrt(3) / lscale_base_og;
    A[1,1] = (1 + dt * lda) * exp(-dt * lda);
    A[1,2] = dt * exp(-dt * lda);
    A[2,1] = -dt * lda * lda * exp(-dt * lda);
    A[2,2] = (1 - dt * lda) * exp(-dt * lda);
    Q[1] = s_base_og * s_base_og * (1 - exp(-2 * dt * lda) * ((1 + dt * lda) * 
            (1 + dt * lda) + dt * lda * dt * lda));
    Q[2] = s_base_og * s_base_og * (lda * lda - exp(-2 * dt * lda) * (lda * lda * 
            (1 - dt * lda) * (1 - dt * lda) + dt * dt * lda * lda * lda *
            lda));
    covQ = 2 * s_base_og * s_base_og * dt * dt * lda * lda * lda * exp(-2 * dt * lda);
     
    x_base_og[1] = mu_og + s_base_og * x_base_shock_og[1];
    m_basederiv = 0; 
    P_basederiv = pow(s_base_og * lda,2);

    for (n in 2:N) {
        m_base = mu_og + A[1,1] * (x_base_og[n-1] - mu_og) + A[1,2] * m_basederiv;
        P_base = Q[1] + A[1,2] * A[1,2] * P_basederiv;
        x_base_og[n] = m_base + sqrt(P_base) * x_base_shock_og[n];
        cov = covQ + A[1,2] * A[2,2] * P_basederiv;
        m_basederiv = A[2,1] * (x_base_og[n-1] - mu_og) + A[2,2] * m_basederiv  +
                       (cov / P_base) * (x_base_og[n] - m_base);
        P_basederiv = A[2,2] * A[2,2] * P_basederiv + Q[2] - 
                       cov * cov / P_base;
    }



    //Constructing x_if
    A_local = exp(-dt / lscale_local_if);
    sqrtQ_local = s_local_if * sqrt(1 - exp(-2 * dt / lscale_local_if));

    for (d in 1:D) {
        x_if[d][1] = x_base_if[1] + s_local_if * x_shock_if[d][1];
        for (i in 2:N) {
            x_if[d][i] = x_base_if[i] + A_local * (x_if[d][i-1] - x_base_if[i-1]) +
                       sqrtQ_local * x_shock_if[d][i];
        }
    }

    //Constructing x_og
    A_local = exp(-dt / lscale_local_og);
    sqrtQ_local = s_local_og * sqrt(1 - exp(-2 * dt / lscale_local_og));

    for (d in 1:D) {
        x_og[d][1] = x_base_og[1] + s_local_og * x_shock_og[d][1];
        for (i in 2:N) {
            x_og[d][i] = x_base_og[i] + A_local * (x_og[d][i-1] - x_base_og[i-1]) +
                       sqrtQ_local * x_shock_og[d][i];
        }
    }

    //Computing p_trip
    for (d in 1:D) {
        for (k in 1:N) {
            real tot_intensity;
            tot_intensity = gamma_if[d][k]*exp(x_if[d][k]) + gamma_og[d][k]*exp(x_og[d][k]);
            p_trip[d,k][3] = exp(-tot_intensity);
            p_trip[d,k][1] = (gamma_if[d][k]*exp(x_if[d][k])/tot_intensity) * (1 - exp(-tot_intensity));
            p_trip[d,k][2] = (gamma_og[d][k]*exp(x_og[d][k])/tot_intensity) * (1 - exp(-tot_intensity));
        }
    }
}


model {
    //MODEL SPECIFICATION
    //Parameter priors
    lscale_local_if    ~ student_t(2, 0, 1); 
    s_local_if         ~ student_t(2, 0, 1); 
    lscale_base_if     ~ student_t(2, 0, 0.25); 
    s_base_if          ~ student_t(2, 0, 1);
    //mu_if            ~ U(-inf, inf)
    //invomega_if      ~ U(0,1)
    lscale_local_og    ~ student_t(2, 0, 1); 
    s_local_og         ~ student_t(2, 0, 1); 
    lscale_base_og     ~ student_t(2, 0, 0.25); 
    s_base_og          ~ student_t(2, 0, 1);
    //mu_og            ~ U(-inf, inf)
    //invomega_og      ~ U(0,1)

    for (d in 1:D) {
        gamma_if[d]    ~ gamma(1/(1/invomega_if - 1), 1/(1/invomega_if - 1) );
        gamma_og[d]    ~ gamma(1/(1/invomega_og - 1), 1/(1/invomega_og - 1) );
    }


    //Intensity processes (handled by the transformation)
    //x_base_if        ~ GP(mu_if, Matérn-3/2-covariance(params: *_base_if))
    //x_if[d]          ~ GP(x_base_if, O--U-covariance(params: *_local_if))
    //x_base_og        ~ GP(mu_og, Matérn-3/2-covariance(params: *_base_og))
    //x_og[d]          ~ GP(x_base_og, O--U-covariance(params: *_local_og))


    //Observation model
    for (d in 1:D) {
        for (k in 1:N) {
            trips[d,k] ~ multinomial(p_trip[d,k]);
        }
    }



    //STAN TRICKERY
    x_base_shock_if ~ normal(0, 1);
    x_base_shock_og ~ normal(0, 1);

    for (d in 1:D) {
 	x_shock_if[d] ~ normal(0, 1);
 	x_shock_og[d] ~ normal(0, 1);
    }      
}
