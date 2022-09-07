#This file contains the building blocks for the objective function. 
import tensorflow_probability as tfp
import tensorflow as tf

#Binomial distribution error model
def Bin_dist(mu_logit,total_cpgs):
        
        pp = tf.math.sigmoid(mu_logit)
        Bin_distr = tfp.distributions.Binomial(total_count=total_cpgs,probs=pp)
        
        return Bin_distr


#Zero inflated Negative Binomial distribution error model
def ZINB_dist(mu_log,theta_log,pi_lat,lib_log):

        tfd = tfp.distributions

        pi = tf.math.sigmoid(pi_lat)
        r_fail = tf.math.exp(lib_log+mu_log+theta_log)
        p_suc = tf.math.sigmoid(theta_log)


        pp = tf.stack([pi,1-pi],axis=-1)
        p = tf.stack([0.00001+0.0*p_suc,p_suc],axis=-1)
        r = tf.stack([0.00001+0.0*r_fail,r_fail],axis=-1)


        ZINB_distr = tfd.MixtureSameFamily(
                                                        mixture_distribution = tfd.Categorical(probs=pp),
                                                        components_distribution = tfd.NegativeBinomial(total_count = r,probs=p)
                                              )
            
        return ZINB_distr
            
#KL divergence
def KL(m1,m2,log_var1,log_var2):

        p1 = tf.divide(tf.square(m1-m2),tf.math.exp(log_var2))
        p2 = tf.divide(tf.math.exp(log_var1),tf.math.exp(log_var2))-1
        p3 = -log_var1+log_var2
        
        return (p1+p2+p3)/2.0

#Jeffrey's divergence
def Jeff(m1,m2,log_var1,log_var2):

        return KL(m1,m2,log_var1,log_var2) + KL(m2,m1,log_var2,log_var1)
