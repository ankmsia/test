data {
  int<lower=1> T;             //number of trials
  int<lower=1> S;             //number of sessions
  int<lower=1> L[S];          //trials per session
  int<lower=0,upper=2> C1[T];  //choice in each trial -- 1 straight, 2 swerve, 0 no choice
  int<lower=0,upper=2> C2[T];  //opponent's choice
  real Vsfe;                  //payoff in front of wall
  real Vstr[T];               //straight payoff
  real Vcop[T];               //cooperative swerve payoff
  real R1[T];                  //recieved reward
  int<lower=0,upper=1> QI;    //gate RL influence on utility
  int<lower=0,upper=1> KI;    //gate choice autocorrelation
  int<lower=0,upper=1> VI;    //gate value
  int<lower=0,upper=1> PI;    //gate swerve prob learning
}

parameters {
  real<lower=0,upper=1> lambda;   //swerve learning rate
  real<lower=0,upper=1> alpha;    //reward learning rate
  real<lower=0,upper=1> tau;      //choice effect decay 
  real<lower=0> Q0;                //initial value
  real<lower=0,upper=1> P0[2];       //initial prob of opponent swerving
  real beta_q[2];                    //RL value weight
  real beta_k[2];                    //choice history weight
  real beta_v[2];                    //Current trial payoff weight
  real beta_coop[2];		    // swerve learning rate based on payoff struc
  real b[2];                         //bias towards swerve
}

transformed parameters {
  real P[T,2];                    //prob of opponent swerving 
  real Q[T,2];                  //RL estimated value
  real K[T,2];                  //perseveration component
  real U[T];                    //utility!
  
  {
    int nxtsess;
    int sess;
    nxtsess = 1;
    sess = 0;

    for (t in 1:T) {
      int TT[t,1];
      TT[t] = Vstr[t]>Vcop[t];

      if (t == nxtsess) { //reset to initial values at new session
        sess = sess + 1;
        nxtsess = nxtsess + L[sess];
        for (i in 1:2) {
          Q[t,i] = Q0;
          K[t,i] = 0;
        }
        P[t] = P0;
        
      } else if (C1[t-1]==0) { //aborted trial
        for (i in 1:2) {
          Q[t,i] = Q[t-1,i];
          K[t,i] = K[t-1,i]*(1-tau);
        }
        P[t] = P[t-1];
        
      } else {
        for (i in 1:2) { //update for successful trials
          Q[t,i] = Q[t-1,i] + ((C1[t-1]==i) ? 
            alpha*(R1[t-1] - Q[t-1,i]) : 0);
          K[t,i] = K[t-1,i]*(1-tau) + 
            ((C1[t-1]==i) ? 1 : 0);
	


        }
        
        P[t,TT] = P[t-1,TT] + lambda*((C2[t-1]-1) - P[t-1,TT])*PI;
      }
	
      //combined utility

      U[t] = beta_v[TT[t]]*( (1-P[t,TT])*Vsfe + P[t]*Vcop[t] - P[t,TT]*Vstr[t])*VI + 
        beta_q[TT[t]]*(Q[t,2]-Q[t,1])*QI + beta_k[TT[t]]*(K[t,2]-K[t,1])*KI + b[TT[t]];
    }
  }
}

model {
  for (t in 1:T)
    if (C1[t]>0) (C1[t]-1) ~ bernoulli_logit(U[t]);
  // P0 ~ beta(2,2);
  // lambda ~ beta(2,2);
  // alpha ~ beta(2,2);
  // tau ~ beta(2,2);
  // beta_q ~ normal(2,10);
  // beta_k ~ normal(2,10);
  // beta_v ~ normal(2,10);
  // b ~ normal(0,10);
  // Q0 ~ normal(1,5);
}

// generated quantities {
//   real loglik;
//   
//   loglik = 0;
//   for (t in 1:T) {
//     if (C1[t]>0) loglik = loglik + bernoulli_logit_lpmf(C[t]-1 | U[t]);
//   }
// }