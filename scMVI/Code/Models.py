#This file contains scMVI models for single omics and multi-omics data.
import tensorflow as tf
from Obj_Blocks import *
from VAE_blocks import Enc_X_Proj,Enc_Man_Proj,Dec_Man_Proj,Dec_X_Proj,Sampling

class VAE(tf.keras.Model):
                def __init__(self,inp_dim,interm_dim,lt_dim,tp,nms,**kwargs):
                                    super(VAE, self).__init__(**kwargs)
                
                                    self.inp_dim = inp_dim
                                    self.interm_dim = interm_dim
                                    self.lt_dim = lt_dim
                                    self.nms = nms
                                    self.tp = tp
                
                                    self.encoder_proj = Enc_X_Proj(inp_sz = self.inp_dim,inter_dim = self.interm_dim , mdl_name= self.nms[0],tp=self.tp).call()
                                    self.encoder_man = Enc_Man_Proj(inp_sz = self.interm_dim,lt_dim = self.lt_dim, mdl_name= self.nms[1]).call()
                                    self.nrm_lay = Sampling()
                                    self.decoder_man = Dec_Man_Proj(lt_dim = self.lt_dim,out_sz = self.interm_dim,mdl_name = self.nms[3]).call()
                                    self.decoder_proj = Dec_X_Proj(inp_sz = self.interm_dim, out_sz = self.inp_dim ,mdl_name = self.nms[4],tp=self.tp).call()
                                    
                                    if self.tp =='tp1':
                                          self.error_mdl = Bin_dist
                                    else:
                                          self.error_mdl = ZINB_dist
                
                def get_latent_params(self,xx):   
                
                                    encoder_proj = self.encoder_proj(xx)
                                    mean,log_var = self.encoder_man(encoder_proj)  
                
                                    return mean,log_var  
                
                def sample_latent(self,xx):
                
                                    return self.nrm_lay(xx)  
                
                def get_latent(self,xx):
                
                                    mean,log_var = self.get_latent_params(xx)
                                    z = self.sample_latent([mean,log_var])
                                    return z  
                                    
                def get_llk_prms(self,xx):
                
                                    r_man = self.decoder_man(xx)
                                    r = self.decoder_proj(r_man)
                                    return r
                    
                def call(self, xx):
                
                                    z_mean, z_log_var = self.get_latent_params(xx)
                                    z = self.sample_latent([z_mean, z_log_var])

                                    r = self.get_llk_prms(z)  
                                  
                                    return r

#scMVI(tp1) for simgle omics data
class scMVI_tp1(tf.keras.Model):
            def __init__(self,inp_tp1_dim,intermidiate_dim,latent_dimensions,**kwargs):
                              super(scMVI_tp1, self).__init__(**kwargs)

                              self.tp1_model_names = ["encoder_proj_tp1","encoder_man_tp1","nrm_lay","decoder_man_tp1","decoder_proj_tp1"]
                              self.inp_tp1_dim = inp_tp1_dim
                              self.intermidiate_dim = intermidiate_dim
                              self.latent_dimensions = latent_dimensions
                              self.tp2_VAE = VAE(inp_dim = self.inp_tp1_dim, interm_dim = self.intermidiate_dim ,lt_dim = self.latent_dimensions ,tp = 'tp1',nms = self.tp1_model_names)
            
            
                              self.reg_KL = KL
                              self.KL_weight = tf.Variable(0.000,trainable=False,dtype=tf.float32)
                              self.total_loss_tracker = tf.keras.metrics.Mean(name="tot.loss")
                              self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
                                      name="rec.loss"
                                  )
                              self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")



            @property
            def metrics(self):
                                  return [
                                      self.total_loss_tracker,
                                      self.reconstruction_loss_tracker,
                                      self.kl_loss_tracker,
                                  ]

            def get_latent_params(self,xx):

                              data_tp1_level,data_tp1_cov = xx
                              z_mean_tp2, z_log_var_tp2 = self.tp1_VAE.get_latent_params([data_tp1_level,data_tp1_cov])

                              return z_mean_tp1, z_log_var_tp1

            def sample_latent(self,xx):

                              z_mean_tp1, z_log_var_tp1 = xx
                              z_tp1 = self.tp1_VAE.nrm_lay([z_mean_tp1, z_log_var_tp1])

                              return z_tp1



            def get_latent(self,xx):

                                z_mean_tp1, z_log_var_tp1  = self.get_latent_params(xx)
                                z_tp1 = self.sample_latent([z_mean_tp1, z_log_var_tp1])

                                return z_tp1



            def tp1_unb_prms(self,xx):

                                  return self.tp1_VAE.get_llk_prms(xx)

            def imp_cells(self,xx):

                          _,data_tp1_cov = xx
                          z_tp1 = self.get_latent(xx)
                          r_tp1 = self.tp1_unb_prms(z_tp1)

                          tp1_smps = self.tp1_VAE.error_mdl(r_tp1,data_tp1_cov).sample()

                          return tp1_smps

            def Diff_Dir_Dist_Avg(self,xx,grp_ind1,grp_ind2,mod):

                          N_samps = 500
                          z_mean_tp2, z_log_var_tp2  = self.get_latent_params(xx)

                          m_tp2_1 = tf.gather(z_mean_tp2,grp_ind1)
                          m_tp2_1 = tf.reduce_mean(m_tp2_1,axis=0)

                          sig_tp2_1 = tf.exp(2.0*tf.gather(z_log_var_tp2,grp_ind1))
                          sig_tp2_1 = tf.math.sqrt(tf.reduce_mean(sig_tp2_1,axis=0))


                          m_tp2_2 = tf.gather(z_mean_tp2,grp_ind2)
                          m_tp2_2 = tf.reduce_mean(m_tp2_2,axis=0)
                          
                          sig_tp2_2 = tf.exp(2.0*tf.gather(z_log_var_tp2,grp_ind2))   
                          sig_tp2_2 = tf.math.sqrt(tf.reduce_mean(sig_tp2_2,axis=0))
                          

                          Normals_tp2_grp1 = tfp.distributions.Normal(loc=m_tp2_1,scale=sig_tp2_1)
                          Normals_tp2_grp2 = tfp.distributions.Normal(loc=m_tp2_2,scale=sig_tp2_2)

                          
                          lt_dir = Normals_tp2_grp1.sample(N_samps) - Normals_tp2_grp2.sample(N_samps)
                          lt_dir_0 = 0.0*lt_dir
                                    
                          diff_dir_r_1 = self.tp1_unb_prms(lt_dir)
                          diff_dir_r_2 = self.tp1_unb_prms(lt_dir_0)
                
                          col_tp1 = diff_dir_r_1-diff_dir_r_2

                          return col_tp1

            def train_step(self, x_tr):

                        with tf.GradientTape() as tape:

                                            data_tp1_level,data_tp1_cov = x_tr

                                            z_mean_tp1, z_log_var_tp1 = self.get_latent_params(x_tr)
                                            z_tp1 = self.sample_latent([z_mean_tp1, z_log_var_tp1])

                                            r_tp1 = self.tp1_unb_prms(z_tp1)


                                            reconstruction_loss_tp1 = tf.reduce_mean(
                                                tf.reduce_sum(
                                                    -self.tp1_VAE.error_mdl(r_tp1,data_tp1_cov).log_prob(data_tp1_level) ,axis=1
                                                )
                                            )

                                            kl_loss_tp1 =  self.reg_KL(z_mean_tp1,0.0*z_mean_tp1,z_log_var_tp1,0.0*z_log_var_tp1) 
                                            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_tp1, axis=1))

                                            reconstruction_loss =  reconstruction_loss_tp1 
                                            total_loss = reconstruction_loss + self.KL_weight*kl_loss
                                          
                        grads = tape.gradient(total_loss, self.trainable_weights)

                        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                        return {
                            "tr.tot.loss": total_loss,
                            "tr.rec.loss": reconstruction_loss,
                            "tr.kl_loss": kl_loss,
                        }


            def call(self, x_tst):
                
                                        data_tp1_level,data_tp1_cov = x_tst

                                        z_mean_tp1, z_log_var_tp1 = self.get_latent_params(x_tst)
                                        z_tp1 = self.sample_latent([z_mean_tp1, z_log_var_tp1])


                                        r_tp1 = self.tp1_unb_prms(z_tp1)
                                        
                                        reconstruction_loss_tp1 = tf.reduce_mean(
                                            tf.reduce_sum(
                                                -self.tp1_VAE.error_mdl(r_tp1,data_tp1_cov).log_prob(data_tp1_level) ,axis=1
                                            )
                                        )

                                        kl_loss_tp1 =  self.reg_KL(z_mean_tp1,0.0*z_mean_tp1,z_log_var_tp1,0.0*z_log_var_tp1) 
                                        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_tp1, axis=1))

                                        reconstruction_loss =  reconstruction_loss_tp1 
                                        total_loss = reconstruction_loss + self.KL_weight*kl_loss
                                        
                                        self.total_loss_tracker.update_state(total_loss)
                                        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                                        self.kl_loss_tracker.update_state(kl_loss)
                                        

                                        return total_loss


#scMVI(tp2) for simgle omics data
class scMVI_tp2(tf.keras.Model):
            def __init__(self,inp_tp2_dim,intermidiate_dim,latent_dimensions,**kwargs):
                              super(scMVI_tp2, self).__init__(**kwargs)

                              self.tp2_model_names = ["encoder_proj_tp2","encoder_man_tp2","nrm_lay","decoder_man_tp2","decoder_proj_tp2"]
                              self.inp_tp2_dim = inp_tp2_dim
                              self.intermidiate_dim = intermidiate_dim
                              self.latent_dimensions = latent_dimensions
                              self.tp2_VAE = VAE(inp_dim = self.inp_tp2_dim, interm_dim = self.intermidiate_dim ,lt_dim = self.latent_dimensions ,tp = 'tp2',nms = self.tp2_model_names)
            
            
                              self.reg_KL = KL
                              self.KL_weight = tf.Variable(0.000,trainable=False,dtype=tf.float32)
                              self.total_loss_tracker = tf.keras.metrics.Mean(name="tot.loss")
                              self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
                                      name="rec.loss"
                                  )
                              self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")



            @property
            def metrics(self):
                                  return [
                                      self.total_loss_tracker,
                                      self.reconstruction_loss_tracker,
                                      self.kl_loss_tracker,
                                  ]

            def get_latent_params(self,xx):

                              data_tp2,_ = xx
                              z_mean_tp2, z_log_var_tp2 = self.tp2_VAE.get_latent_params(data_tp2)

                              return z_mean_tp2, z_log_var_tp2

            def sample_latent(self,xx):

                              z_mean_tp2, z_log_var_tp2 = xx
                              z_tp2 = self.tp2_VAE.nrm_lay([z_mean_tp2, z_log_var_tp2])

                              return z_tp2



            def get_latent(self,xx):

                                z_mean_tp2, z_log_var_tp2  = self.get_latent_params(xx)
                                z_tp2 = self.sample_latent([z_mean_tp2, z_log_var_tp2])

                                return z_tp2



            def tp2_unb_prms(self,xx):

                                  return self.tp2_VAE.get_llk_prms(xx)

            def imp_cells(self,xx):

                          _,lib_z_tp2 = xx
                          z_tp2 = self.get_latent(xx)
                          
                          r_tp2_r,r_tp2_theta, r_tp2_drop = self.tp2_unb_prms(z_tp2)

                          tp2_smps = self.tp2_VAE.error_mdl(r_tp2_r,r_tp2_theta,r_tp2_drop,lib_z_tp2).sample()

                          return tp2_smps

            def Diff_Dir_Dist_Avg(self,xx,grp_ind1,grp_ind2,mod):

                          N_samps = 500
                          z_mean_tp2, z_log_var_tp2  = self.get_latent_params(xx)

                          m_tp2_1 = tf.gather(z_mean_tp2,grp_ind1)
                          m_tp2_1 = tf.reduce_mean(m_tp2_1,axis=0)

                          sig_tp2_1 = tf.exp(2.0*tf.gather(z_log_var_tp2,grp_ind1))
                          sig_tp2_1 = tf.math.sqrt(tf.reduce_mean(sig_tp2_1,axis=0))


                          m_tp2_2 = tf.gather(z_mean_tp2,grp_ind2)
                          m_tp2_2 = tf.reduce_mean(m_tp2_2,axis=0)
                          
                          sig_tp2_2 = tf.exp(2.0*tf.gather(z_log_var_tp2,grp_ind2))   
                          sig_tp2_2 = tf.math.sqrt(tf.reduce_mean(sig_tp2_2,axis=0))
                          

                          Normals_tp2_grp1 = tfp.distributions.Normal(loc=m_tp2_1,scale=sig_tp2_1)
                          Normals_tp2_grp2 = tfp.distributions.Normal(loc=m_tp2_2,scale=sig_tp2_2)

                          
                          lt_dir = Normals_tp2_grp1.sample(N_samps) - Normals_tp2_grp2.sample(N_samps)
                          lt_dir_0 = 0.0*lt_dir
                                    
                          diff_dir_r_1,diff_dir_theta_1,_ = self.tp2_unb_prms(lt_dir)
                          diff_dir_r_2,diff_dir_theta_2,_ = self.tp2_unb_prms(lt_dir_0)
                
                          col_tp2 = diff_dir_r_1+diff_dir_theta_1-(diff_dir_r_2+diff_dir_theta_2)

                          return col_tp2

            def train_step(self, x_tr):

                        with tf.GradientTape() as tape:

                                            data_tp2,lib_z_tp2 = x_tr

                                            z_mean_tp2, z_log_var_tp2 = self.get_latent_params(x_tr)
                                            z_tp2 = self.sample_latent([z_mean_tp2, z_log_var_tp2])

                                            r_tp2_r,r_tp2_theta, r_tp2_drop = self.tp2_unb_prms(z_tp2)


                                            reconstruction_loss_tp2 = tf.reduce_mean(
                                                tf.reduce_sum(
                                                    -self.tp2_VAE.error_mdl(r_tp2_r,r_tp2_theta,r_tp2_drop,lib_z_tp2).log_prob(data_tp2) ,axis=1
                                                )
                                            )

                                            kl_loss_tp2 =  self.reg_KL(z_mean_tp2,0.0*z_mean_tp2,z_log_var_tp2,0.0*z_log_var_tp2) 
                                            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_tp2, axis=1))

                                            reconstruction_loss =  reconstruction_loss_tp2 
                                            total_loss = reconstruction_loss + self.KL_weight*kl_loss
                                          
                        grads = tape.gradient(total_loss, self.trainable_weights)

                        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                        return {
                            "tr.tot.loss": total_loss,
                            "tr.rec.loss": reconstruction_loss,
                            "tr.kl_loss": kl_loss,
                        }


            def call(self, x_tst):
                
                                        data_tp2,lib_z_tp2 = x_tst

                                        z_mean_tp2, z_log_var_tp2 = self.get_latent_params(x_tst)
                                        z_tp2 = self.sample_latent([z_mean_tp2, z_log_var_tp2])


                                        r_tp2_r,r_tp2_theta, r_tp2_drop = self.tp2_unb_prms(z_tp2)
                                        reconstruction_loss_tp2 = tf.reduce_mean(
                                            tf.reduce_sum(
                                                -self.tp2_VAE.error_mdl(r_tp2_r,r_tp2_theta,r_tp2_drop,lib_z_tp2).log_prob(data_tp2) ,axis=1
                                            )
                                        )

                                        kl_loss_tp2 =  self.reg_KL(z_mean_tp2,0.0*z_mean_tp2,z_log_var_tp2,0.0*z_log_var_tp2) 
                                        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_tp2, axis=1))

                                        reconstruction_loss =  reconstruction_loss_tp2 
                                        total_loss = reconstruction_loss + self.KL_weight*kl_loss
                                        
                                        self.total_loss_tracker.update_state(total_loss)
                                        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                                        self.kl_loss_tracker.update_state(kl_loss)
                                        

                                        return total_loss



#scMVI for NMT data
class scMVI_NMT(tf.keras.Model):
            def __init__(self,inp_met_dim,inp_acc_dim,inp_rna_dim,intermidiate_dim,latent_dimensions,**kwargs):
                              super(scMVI_NMT, self).__init__(**kwargs)

                              self.met_model_names = ["encoder_proj_met","encoder_man_met","nrm_lay","decoder_man_met","decoder_proj_met"]
                              self.acc_model_names = ["encoder_proj_acc","encoder_man_acc","nrm_lay","decoder_man_acc","decoder_proj_acc"]
                              self.rna_model_names = ["encoder_proj_rna","encoder_man_rna","nrm_lay","decoder_man_rna","decoder_proj_rna"]

                              self.inp_met_dim = inp_met_dim
                              self.inp_acc_dim = inp_acc_dim
                              self.inp_rna_dim = inp_rna_dim
                              self.intermidiate_dim = intermidiate_dim
                              self.latent_dimensions = latent_dimensions

                              self.met_VAE = VAE(inp_dim = self.inp_met_dim, interm_dim = self.intermidiate_dim ,lt_dim = self.latent_dimensions ,tp = 'tp1',nms = self.met_model_names)
                              self.acc_VAE = VAE(inp_dim = self.inp_acc_dim, interm_dim = self.intermidiate_dim ,lt_dim = self.latent_dimensions ,tp = 'tp1',nms = self.acc_model_names)
                              self.rna_VAE = VAE(inp_dim = self.inp_rna_dim, interm_dim = self.intermidiate_dim ,lt_dim = self.latent_dimensions ,tp = 'tp2',nms = self.rna_model_names)
            
            
                              self.reg_jeff = Jeff
                              self.reg_KL = KL
                              self.KL_weight = tf.Variable(0.000,trainable=False,dtype=tf.float32)
                              self.total_loss_tracker = tf.keras.metrics.Mean(name="tot.loss")
                              self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
                                      name="rec.loss"
                                  )
                              self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

            def get_config(self):
                                  config = super(scMVI_NMT, self).get_config()
                                  config.update({"inp_met": self.inp_met,"inp_acc": self.inp_acc,"inp_rna": self.inp_rna,"intermidiate_dim": self.intermidiate_dim , "latent_dimensions":self.latent_dimensions})
                                  return config


            @property
            def metrics(self):
                                  return [
                                      self.total_loss_tracker,
                                      self.reconstruction_loss_tracker,
                                      self.kl_loss_tracker,
                                  ]

            def get_latent_params(self,xx):

                              data_met,data_cpg,data_acc,data_gpc,data_rna,_ = xx

                              z_mean_met, z_log_var_met = self.met_VAE.get_latent_params([data_met,data_cpg])
                              z_mean_acc, z_log_var_acc = self.acc_VAE.get_latent_params([data_acc,data_gpc])
                              z_mean_rna, z_log_var_rna = self.rna_VAE.get_latent_params(data_rna)

                              return z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna 

            def sample_latent(self,xx):

                              z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna = xx

                              z_met = self.met_VAE.nrm_lay([z_mean_met, z_log_var_met])
                              z_acc = self.acc_VAE.nrm_lay([z_mean_acc, z_log_var_acc])
                              z_rna = self.rna_VAE.nrm_lay([z_mean_rna, z_log_var_rna])

                              return z_met,z_acc,z_rna 



            def get_latent(self,xx):

                                z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna  = self.get_latent_params(xx)
                                z_met,z_acc,z_rna = self.sample_latent([z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna])

                                return z_met,z_acc,z_rna

            @staticmethod
            def comb_lt(xx,lab):

                                  z_met,z_acc,z_rna = xx
                            
                                  if lab =='all':
                                        z = (z_acc+z_met+z_rna)/3.0
                                  elif lab == 'met-acc':
                                        z = (z_acc+z_met)/2.0
                                  elif lab == 'met-rna':
                                        z = (z_rna+z_met)/2.0
                                  elif lab == 'acc-rna':
                                        z = (z_acc+z_rna)/2.0
                                  elif lab == 'met':
                                        z = z_met
                                  elif lab == 'acc':
                                        z = z_acc
                                  elif lab == 'rna':
                                        z = z_rna                                       
                                  else:
                                        raise ValueError('Please choose a correct label! Correct labels are all, meth-acc, meth-rna, acc-rna., met, acc, rna')    
                                        
                                  return z 


            def met_unb_prms(self,xx):

                                  return self.met_VAE.get_llk_prms(xx)

            def acc_unb_prms(self,xx):

                                  return self.acc_VAE.get_llk_prms(xx)

            def rna_unb_prms(self,xx):

                                  return self.rna_VAE.get_llk_prms(xx)

            def imp_cells(self,xx,mod):
                          # need to find a way for choosing scaling factors.
                          _,data_cpg,_,data_gpc,_,lib_z_rna = xx
                          z_met,z_acc,z_rna = self.get_latent(xx)
                          z = scMVI_NMT.comb_lt([z_met,z_acc,z_rna],mod)

                          r_met = self.met_unb_prms(z) 
                          r_acc = self.acc_unb_prms(z)  
                          r_rna_r,r_rna_theta, r_rna_drop = self.rna_unb_prms(z)

                          met_smps = self.met_VAE.error_mdl(r_met,data_cpg).sample()
                          acc_smps = self.acc_VAE.error_mdl(r_acc,data_gpc).sample()
                          rna_smps = self.rna_VAE.error_mdl(r_rna_r,r_rna_theta,r_rna_drop,lib_z_rna).sample()

                          return met_smps,acc_smps, rna_smps

            def Diff_Dir_Dist_Avg(self,xx,grp_ind1,grp_ind2,mod):

                          N_samps = 500
                          z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna  = self.get_latent_params(xx)

                          m_met_1 = tf.gather(z_mean_met,grp_ind1)
                          m_met_1 = tf.reduce_mean(m_met_1,axis=0)

                          sig_met_1 = tf.exp(2.0*tf.gather(z_log_var_met,grp_ind1))
                          sig_met_1 = tf.math.sqrt(tf.reduce_mean(sig_met_1,axis=0))

                          m_acc_1 = tf.gather(z_mean_acc,grp_ind1)
                          m_acc_1 = tf.reduce_mean(m_acc_1,axis=0)

                          sig_acc_1 = tf.exp(2.0*tf.gather(z_log_var_acc,grp_ind1))
                          sig_acc_1 = tf.math.sqrt(tf.reduce_mean(sig_acc_1,axis=0))

                          m_rna_1 = tf.gather(z_mean_rna,grp_ind1)
                          m_rna_1 = tf.reduce_mean(m_rna_1,axis=0)

                          sig_rna_1 = tf.exp(2.0*tf.gather(z_log_var_rna,grp_ind1))
                          sig_rna_1 = tf.math.sqrt(tf.reduce_mean(sig_rna_1,axis=0))

                          m_met_2 = tf.gather(z_mean_met,grp_ind2)
                          m_met_2 = tf.reduce_mean(m_met_2,axis=0)

                          sig_met_2 = tf.exp(2.0*tf.gather(z_log_var_met,grp_ind2))
                          sig_met_2 = tf.math.sqrt(tf.reduce_mean(sig_met_2,axis=0))

                          m_acc_2 = tf.gather(z_mean_acc,grp_ind2)
                          m_acc_2 = tf.reduce_mean(m_acc_2,axis=0)

                          sig_acc_2 = tf.exp(2.0*tf.gather(z_log_var_acc,grp_ind2))
                          sig_acc_2 = tf.math.sqrt(tf.reduce_mean(sig_acc_2,axis=0))

                          m_rna_2 = tf.gather(z_mean_rna,grp_ind2)
                          m_rna_2 = tf.reduce_mean(m_rna_2,axis=0)
                          
                          sig_rna_2 = tf.exp(2.0*tf.gather(z_log_var_rna,grp_ind2))   
                          sig_rna_2 = tf.math.sqrt(tf.reduce_mean(sig_rna_2,axis=0))
                          

                          Normals_rna_grp1 = tfp.distributions.Normal(loc=m_rna_1,scale=sig_rna_1)
                          Normals_rna_grp2 = tfp.distributions.Normal(loc=m_rna_2,scale=sig_rna_2)

                          Normals_acc_grp1 = tfp.distributions.Normal(loc=m_acc_1,scale=sig_acc_1)
                          Normals_acc_grp2 = tfp.distributions.Normal(loc=m_acc_2,scale=sig_acc_2)

                          Normals_met_grp1 = tfp.distributions.Normal(loc=m_met_1,scale=sig_met_1)
                          Normals_met_grp2 = tfp.distributions.Normal(loc=m_met_2,scale=sig_met_2)


                          lt_dir = scMVI_NMT.comb_lt([Normals_met_grp1.sample(N_samps),Normals_acc_grp1.sample(N_samps),Normals_rna_grp1.sample(N_samps)],mod) - scMVI_NMT.comb_lt([Normals_met_grp2.sample(N_samps),Normals_acc_grp2.sample(N_samps),Normals_rna_grp2.sample(N_samps)],mod)#
                          lt_dir_0 = 0.0*lt_dir
                                    
                          rna_diff_dir_r_1,rna_diff_dir_theta_1,_ = self.rna_unb_prms(lt_dir)
                          rna_diff_dir_r_2,rna_diff_dir_theta_2,_ = self.rna_unb_prms(lt_dir_0)

                          acc_diff_dir1 = self.acc_unb_prms(lt_dir)
                          acc_diff_dir2 = self.acc_unb_prms(lt_dir_0)                          

                          met_diff_dir1 = self.met_unb_prms(lt_dir)
                          met_diff_dir2 = self.met_unb_prms(lt_dir_0)     

                          col_met = met_diff_dir1-met_diff_dir2
                          col_acc = acc_diff_dir1-acc_diff_dir2
                          col_rna = rna_diff_dir_r_1+rna_diff_dir_theta_1-(rna_diff_dir_r_2+rna_diff_dir_theta_2)

                          return col_met,col_acc,col_rna



            def train_step(self, x_tr):

                        with tf.GradientTape() as tape:
                                            data_met,data_cpg,data_acc,data_gpc,data_rna,lib_z_rna = x_tr

                                            z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna = self.get_latent_params(x_tr)
                                            z_met,z_acc,z_rna = self.sample_latent([z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna])


                                            r_met = self.met_unb_prms(z_met)  
                                            r_acc = self.acc_unb_prms(z_acc)  
                                            r_rna_r,r_rna_theta, r_rna_drop = self.rna_unb_prms(z_rna)

                                            reconstruction_loss_met = tf.reduce_mean(
                                                tf.reduce_sum( -self.met_VAE.error_mdl(r_met,data_cpg).log_prob(data_met) ,axis=1
                                                )
                                            )

                                            reconstruction_loss_acc = tf.reduce_mean(
                                                tf.reduce_sum( -self.acc_VAE.error_mdl(r_acc,data_gpc).log_prob(data_acc)  ,axis=1
                                                )
                                            )
                                            reconstruction_loss_rna = tf.reduce_mean(
                                                tf.reduce_sum(
                                                    -self.rna_VAE.error_mdl(r_rna_r,r_rna_theta,r_rna_drop,lib_z_rna).log_prob(data_rna) ,axis=1
                                                )
                                            )


                                            kl_loss_met =  self.reg_jeff(z_mean_met,z_mean_rna,z_log_var_met,z_log_var_rna) + self.KL_weight*self.reg_KL(z_mean_met,0*z_mean_met,z_log_var_met,0*z_log_var_met) 
                                            kl_loss_acc =  self.reg_jeff(z_mean_acc,z_mean_rna,z_log_var_acc,z_log_var_rna) + self.KL_weight*self.reg_KL(z_mean_acc,0*z_mean_acc,z_log_var_acc,0*z_log_var_acc) 
                                            kl_loss_rna =  self.reg_jeff(z_mean_rna,z_mean_met,z_log_var_rna,z_log_var_met) + self.KL_weight*self.reg_KL(z_mean_rna,0*z_mean_rna,z_log_var_rna,0*z_log_var_rna) 
                                            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_rna, axis=1)) + tf.reduce_mean(tf.reduce_sum(kl_loss_met, axis=1)) + tf.reduce_mean(tf.reduce_sum(kl_loss_acc, axis=1))

                                            reconstruction_loss =  reconstruction_loss_rna + reconstruction_loss_met + reconstruction_loss_acc
                                            total_loss = reconstruction_loss + kl_loss
                                          
                        grads = tape.gradient(total_loss, self.trainable_weights)

                        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                        return {
                            "tr.tot.loss": total_loss,
                            "tr.rec.loss": reconstruction_loss,
                            "tr.kl_loss": kl_loss,
                        }


            def call(self, x_tst):
                
                                        data_met,data_cpg,data_acc,data_gpc,data_rna,lib_z_rna = x_tst

                                        z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna = self.get_latent_params(x_tst)
                                        z_met,z_acc,z_rna = self.sample_latent([z_mean_met, z_log_var_met,z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna])


                                        r_met = self.met_unb_prms(z_met)  
                                        r_acc = self.acc_unb_prms(z_acc)  
                                        r_rna_r,r_rna_theta, r_rna_drop = self.rna_unb_prms(z_rna)

                                        reconstruction_loss_met = tf.reduce_mean(
                                            tf.reduce_sum( -self.met_VAE.error_mdl(r_met,data_cpg).log_prob(data_met) ,axis=1
                                            )
                                        )

                                        reconstruction_loss_acc = tf.reduce_mean(
                                            tf.reduce_sum( -self.acc_VAE.error_mdl(r_acc,data_gpc).log_prob(data_acc)  ,axis=1
                                            )
                                        )
                                        reconstruction_loss_rna = tf.reduce_mean(
                                            tf.reduce_sum(
                                                -self.rna_VAE.error_mdl(r_rna_r,r_rna_theta,r_rna_drop,lib_z_rna).log_prob(data_rna) ,axis=1
                                            )
                                        )


                                        kl_loss_met =  self.reg_jeff(z_mean_met,z_mean_rna,z_log_var_met,z_log_var_rna) + self.KL_weight*self.reg_KL(z_mean_met,0*z_mean_met,z_log_var_met,0*z_log_var_met) 
                                        kl_loss_acc =  self.reg_jeff(z_mean_acc,z_mean_rna,z_log_var_acc,z_log_var_rna) + self.KL_weight*self.reg_KL(z_mean_acc,0*z_mean_acc,z_log_var_acc,0*z_log_var_acc) 
                                        kl_loss_rna =  self.reg_jeff(z_mean_rna,z_mean_met,z_log_var_rna,z_log_var_met) + self.KL_weight*self.reg_KL(z_mean_rna,0*z_mean_rna,z_log_var_rna,0*z_log_var_rna) 
                                        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_rna, axis=1)) + tf.reduce_mean(tf.reduce_sum(kl_loss_met, axis=1)) + tf.reduce_mean(tf.reduce_sum(kl_loss_acc, axis=1))

                                        reconstruction_loss =  reconstruction_loss_rna + reconstruction_loss_met + reconstruction_loss_acc
                                        total_loss = reconstruction_loss + kl_loss

                                        self.total_loss_tracker.update_state(total_loss)
                                        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                                        self.kl_loss_tracker.update_state(kl_loss)
                                        

                                        return total_loss

#scMVI for 10X multi-omics data
class scMVI_10X(tf.keras.Model):
            def __init__(self,inp_acc_dim,inp_rna_dim,intermidiate_dim,latent_dimensions,**kwargs):
                              super(scMVI_10X, self).__init__(**kwargs)

                              self.acc_model_names = ["encoder_proj_acc","encoder_man_acc","nrm_lay","decoder_man_acc","decoder_proj_acc"]
                              self.rna_model_names = ["encoder_proj_rna","encoder_man_rna","nrm_lay","decoder_man_rna","decoder_proj_rna"]

                              self.inp_acc_dim = inp_acc_dim
                              self.inp_rna_dim = inp_rna_dim
                              self.intermidiate_dim = intermidiate_dim
                              self.latent_dimensions = latent_dimensions

                              self.acc_VAE = VAE(inp_dim = self.inp_acc_dim, interm_dim = self.intermidiate_dim ,lt_dim = self.latent_dimensions ,tp = 'tp2',nms = self.acc_model_names)
                              self.rna_VAE = VAE(inp_dim = self.inp_rna_dim, interm_dim = self.intermidiate_dim ,lt_dim = self.latent_dimensions ,tp = 'tp2',nms = self.rna_model_names)
            
            
                              self.reg_jeff = Jeff
                              self.reg_KL = KL
                              self.KL_weight = tf.Variable(0.000,trainable=False,dtype=tf.float32)
                              self.total_loss_tracker = tf.keras.metrics.Mean(name="tot.loss")
                              self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
                                      name="rec.loss"
                                  )
                              self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

            def get_config(self):
                                  config = super(scMVI_10X, self).get_config()
                                  config.update({"inp_acc": self.inp_acc,"inp_rna": self.inp_rna,"intermidiate_dim": self.intermidiate_dim , "latent_dimensions":self.latent_dimensions})
                                  return config


            @property
            def metrics(self):
                                  return [
                                      self.total_loss_tracker,
                                      self.reconstruction_loss_tracker,
                                      self.kl_loss_tracker,
                                  ]

            def get_latent_params(self,xx):

                              data_acc,_,data_rna,_ = xx

                              z_mean_acc, z_log_var_acc = self.acc_VAE.get_latent_params(data_acc)
                              z_mean_rna, z_log_var_rna = self.rna_VAE.get_latent_params(data_rna)

                              return z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna 

            def sample_latent(self,xx):

                              z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna = xx

                              z_acc = self.acc_VAE.nrm_lay([z_mean_acc, z_log_var_acc])
                              z_rna = self.rna_VAE.nrm_lay([z_mean_rna, z_log_var_rna])

                              return z_acc,z_rna 



            def get_latent(self,xx):

                                z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna  = self.get_latent_params(xx)
                                z_acc,z_rna = self.sample_latent([z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna])

                                return z_acc,z_rna

            @staticmethod
            def comb_lt(xx,lab):

                            z_acc,z_rna = xx
                            if lab =='all':
                                    z = (z_acc+z_rna)/2.0
                                    
                            elif lab == 'rna':
                                    
                                    z = z_rna

                            elif lab == 'acc':
                                    
                                    z = z_acc
                            else:
                                raise ValueError('Please choose a correct label! Correct labels are all, acc, rna')                                       
                                  
                            return z 


            def acc_unb_prms(self,xx):

                                  return self.acc_VAE.get_llk_prms(xx)

            def rna_unb_prms(self,xx):

                                  return self.rna_VAE.get_llk_prms(xx)

            def imp_cells(self,xx,mod):
                          # need to find a way for choosing scaling factors.
                          _,lib_z_acc,_,lib_z_rna = xx
                          z_acc,z_rna = self.get_latent(xx)
                          z = scMVI_10X.comb_lt([z_acc,z_rna],mod)
                          
                          r_acc_r,r_acc_theta, r_acc_drop = self.acc_unb_prms(z)  
                          r_rna_r,r_rna_theta, r_rna_drop = self.rna_unb_prms(z)

                          acc_smps = self.acc_VAE.error_mdl(r_acc_r,r_acc_theta,r_acc_drop,lib_z_acc).sample()
                          rna_smps = self.rna_VAE.error_mdl(r_rna_r,r_rna_theta,r_rna_drop,lib_z_rna).sample()

                          return acc_smps, rna_smps

            def Diff_Dir_Dist_Avg(self,xx,grp_ind1,grp_ind2,mod):

                          N_samps = 500
                          z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna  = self.get_latent_params(xx)

                          m_acc_1 = tf.gather(z_mean_acc,grp_ind1)
                          m_acc_1 = tf.reduce_mean(m_acc_1,axis=0)

                          sig_acc_1 = tf.exp(2.0*tf.gather(z_log_var_acc,grp_ind1))
                          sig_acc_1 = tf.math.sqrt(tf.reduce_mean(sig_acc_1,axis=0))

                          m_rna_1 = tf.gather(z_mean_rna,grp_ind1)
                          m_rna_1 = tf.reduce_mean(m_rna_1,axis=0)

                          sig_rna_1 = tf.exp(2.0*tf.gather(z_log_var_rna,grp_ind1))
                          sig_rna_1 = tf.math.sqrt(tf.reduce_mean(sig_rna_1,axis=0))

                          m_acc_2 = tf.gather(z_mean_acc,grp_ind2)
                          m_acc_2 = tf.reduce_mean(m_acc_2,axis=0)

                          sig_acc_2 = tf.exp(2.0*tf.gather(z_log_var_acc,grp_ind2))
                          sig_acc_2 = tf.math.sqrt(tf.reduce_mean(sig_acc_2,axis=0))

                          m_rna_2 = tf.gather(z_mean_rna,grp_ind2)
                          m_rna_2 = tf.reduce_mean(m_rna_2,axis=0)
                          
                          sig_rna_2 = tf.exp(2.0*tf.gather(z_log_var_rna,grp_ind2))   
                          sig_rna_2 = tf.math.sqrt(tf.reduce_mean(sig_rna_2,axis=0))
                          

                          Normals_rna_grp1 = tfp.distributions.Normal(loc=m_rna_1,scale=sig_rna_1)
                          Normals_rna_grp2 = tfp.distributions.Normal(loc=m_rna_2,scale=sig_rna_2)

                          Normals_acc_grp1 = tfp.distributions.Normal(loc=m_acc_1,scale=sig_acc_1)
                          Normals_acc_grp2 = tfp.distributions.Normal(loc=m_acc_2,scale=sig_acc_2)
                          
                          lt_dir = scMVI_10X.comb_lt([Normals_acc_grp1.sample(N_samps),Normals_rna_grp1.sample(N_samps)],mod) - scMVI_10X.comb_lt([Normals_acc_grp2.sample(N_samps),Normals_rna_grp2.sample(N_samps)],mod)#
                          lt_dir_0 = 0.0*lt_dir
                                    
                          rna_diff_dir_r_1,rna_diff_dir_theta_1,_ = self.rna_unb_prms(lt_dir)
                          rna_diff_dir_r_2,rna_diff_dir_theta_2,_ = self.rna_unb_prms(lt_dir_0)

                          acc_diff_dir_r_1,acc_diff_dir_theta_1,_ = self.acc_unb_prms(lt_dir)
                          acc_diff_dir_r_2,acc_diff_dir_theta_2,_ = self.acc_unb_prms(lt_dir_0)                          

                          col_acc = acc_diff_dir_r_1+acc_diff_dir_theta_1-(acc_diff_dir_r_2+acc_diff_dir_theta_2)
                          col_rna = rna_diff_dir_r_1+rna_diff_dir_theta_1-(rna_diff_dir_r_2+rna_diff_dir_theta_2)

                          return col_acc,col_rna


            def train_step(self, x_tr):

                        with tf.GradientTape() as tape:

                                            data_acc,lib_z_acc,data_rna,lib_z_rna = x_tr

                                            z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna = self.get_latent_params(x_tr)
                                            z_acc,z_rna = self.sample_latent([z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna])


                                            r_acc_r,r_acc_theta, r_acc_drop = self.acc_unb_prms(z_acc)  
                                            r_rna_r,r_rna_theta, r_rna_drop = self.rna_unb_prms(z_rna)


                                            reconstruction_loss_acc = tf.reduce_mean(
                                                tf.reduce_sum(
                                                    -self.acc_VAE.error_mdl(r_acc_r,r_acc_theta,r_acc_drop,lib_z_acc).log_prob(data_acc) ,axis=1
                                                )
                                            )
                                            reconstruction_loss_rna = tf.reduce_mean(
                                                tf.reduce_sum(
                                                    -self.rna_VAE.error_mdl(r_rna_r,r_rna_theta,r_rna_drop,lib_z_rna).log_prob(data_rna) ,axis=1
                                                )
                                            )


                                            kl_loss_acc =  self.reg_jeff(z_mean_acc,z_mean_rna,z_log_var_acc,z_log_var_rna) + self.KL_weight*self.reg_KL(z_mean_acc,0*z_mean_acc,z_log_var_acc,0*z_log_var_acc) 
                                            kl_loss_rna =  self.KL_weight*self.reg_KL(z_mean_rna,0*z_mean_rna,z_log_var_rna,0*z_log_var_rna) 
                                            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_rna, axis=1)) + tf.reduce_mean(tf.reduce_sum(kl_loss_acc, axis=1))

                                            reconstruction_loss =  reconstruction_loss_rna + reconstruction_loss_acc
                                            total_loss = reconstruction_loss + self.KL_weight*kl_loss
                                          
                        grads = tape.gradient(total_loss, self.trainable_weights)

                        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                        return {
                            "tr.tot.loss": total_loss,
                            "tr.rec.loss": reconstruction_loss,
                            "tr.kl_loss": kl_loss,
                        }


            def call(self, x_tst):
                
                                        data_acc,lib_z_acc,data_rna,lib_z_rna = x_tst

                                        z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna = self.get_latent_params(x_tst)
                                        z_acc,z_rna = self.sample_latent([z_mean_acc, z_log_var_acc,z_mean_rna, z_log_var_rna])


                                        r_acc_r,r_acc_theta, r_acc_drop = self.acc_unb_prms(z_acc)  
                                        r_rna_r,r_rna_theta, r_rna_drop = self.rna_unb_prms(z_rna)


                                        reconstruction_loss_acc = tf.reduce_mean(
                                            tf.reduce_sum(
                                                -self.acc_VAE.error_mdl(r_acc_r,r_acc_theta,r_acc_drop,lib_z_acc).log_prob(data_acc) ,axis=1
                                            )
                                        )

                                        reconstruction_loss_rna = tf.reduce_mean(
                                            tf.reduce_sum(
                                                -self.rna_VAE.error_mdl(r_rna_r,r_rna_theta,r_rna_drop,lib_z_rna).log_prob(data_rna) ,axis=1
                                            )
                                        )


                                        kl_loss_acc =  self.reg_jeff(z_mean_acc,z_mean_rna,z_log_var_acc,z_log_var_rna) + self.KL_weight*self.reg_KL(z_mean_acc,0*z_mean_acc,z_log_var_acc,0*z_log_var_acc) 
                                        kl_loss_rna =  self.KL_weight*self.reg_KL(z_mean_rna,0*z_mean_rna,z_log_var_rna,0*z_log_var_rna) 
                                        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss_rna, axis=1)) + tf.reduce_mean(tf.reduce_sum(kl_loss_acc, axis=1))

                                        reconstruction_loss =  reconstruction_loss_rna +  reconstruction_loss_acc
                                        total_loss = reconstruction_loss + kl_loss

                                        self.total_loss_tracker.update_state(total_loss)
                                        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                                        self.kl_loss_tracker.update_state(kl_loss)
                                        

                                        return total_loss