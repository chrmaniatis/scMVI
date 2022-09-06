#This file contains the building blocks of scMVI encoders and decoders.
import tensorflow as tf

class Enc_X_Proj(tf.keras.Model):
            def __init__(self,inp_sz,inter_dim,mdl_name,tp, **kwargs):
                      super(Enc_X_Proj, self).__init__(**kwargs)       
            
                      #Projecting methylation,expression and accessibility to k dimensional manifolds
            
                      self.inp_sz = inp_sz
                      self.inter_dim = inter_dim
                      self.mdl_name = mdl_name
                      self.tp = tp
            
                      if self.tp == 'tp1':
                              
                              self.mdl_imp = [tf.keras.Input(shape=(self.inp_sz,)),tf.keras.Input(shape=(self.inp_sz,))]
                              self.proj_lay = tf.keras.layers.Concatenate()(self.mdl_imp)
                              
                      elif self.tp == 'tp2':
            
                              self.mdl_imp = tf.keras.Input(shape=(self.inp_sz,))  
                              self.proj_lay = self.mdl_imp
            
                      else:
                              raise ValueError(
                                                "tp label accepts two options tp1/tp2"
                                                )      
            
            def call(self):
            
            
                      x_proj = tf.keras.layers.Dense(2*self.inter_dim,activation='linear')(self.proj_lay)
                      x_proj = tf.keras.layers.LayerNormalization(axis=-1)(x_proj)    
                      x_proj = tf.keras.activations.relu(x_proj)          
                      x_proj = tf.keras.layers.Dropout(0.1)(x_proj)
                      x_proj = tf.keras.layers.Dense(self.inter_dim,activation='linear')(x_proj)
                      x_proj = tf.keras.layers.LayerNormalization(axis=-1)(x_proj)    
                      x_proj = tf.keras.activations.relu(x_proj)          
                      x_proj = tf.keras.layers.Dropout(0.1)(x_proj)
            
                      encoder_proj = tf.keras.Model(inputs = self.mdl_imp , outputs=x_proj, name=self.mdl_name)
            
                      return encoder_proj
            

class Enc_Man_Proj(tf.keras.Model):
                def __init__(self,inp_sz,lt_dim,mdl_name, **kwargs):
                          super(Enc_Man_Proj, self).__init__(**kwargs)       
                
                          #Expression, methylation and accessibility lower dimension manifolds
                
                          self.inp_sz = inp_sz
                          self.mdl_name = mdl_name
                          self.lt_dim = lt_dim
                          self.mdl_imp = tf.keras.Input(shape=(self.inp_sz,))  
                
                def call(self):
                
                        x = tf.keras.layers.Dense(self.inp_sz,activation='linear')(self.mdl_imp)
                        x = tf.keras.layers.LayerNormalization(axis=-1)(x)        
                        x = tf.keras.activations.relu(x)                          
                        x = tf.keras.layers.Dropout(0.1)(x)
                        z_mean = tf.keras.layers.Dense(self.lt_dim)(x)
                        z_log_var = tf.keras.layers.Dense(self.lt_dim)(x)
                        mdl = tf.keras.Model(inputs = self.mdl_imp, outputs=[z_mean, z_log_var], name=self.mdl_name)
                
                        return mdl


class Sampling(tf.keras.layers.Layer):
                """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
                def __init__(self, name=None, **kwargs):
                        super(Sampling, self).__init__(name=name)
                        super(Sampling, self).__init__(**kwargs)
                
                def get_config(self):
                        config = super(Sampling, self).get_config()
                        return config
                
                def call(self, inputs):
                        z_mean, z_log_var = inputs
                        batch = tf.shape(z_mean)[0]
                        dim = tf.shape(z_mean)[1]
                        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
                



class Dec_Man_Proj(tf.keras.Model):
    
            def __init__(self,lt_dim,out_sz,mdl_name, **kwargs):
                      super(Dec_Man_Proj, self).__init__(**kwargs)   
            
                      #Reconstruct expression, methylation and accessibility low dimensional manifolds from latent normal samples
                      
                      self.lt_dim = lt_dim
                      self.out_sz = out_sz
                      self.mdl_name = mdl_name
                      self.mdl_imp = tf.keras.Input(shape=(self.lt_dim,))  
            
            def call(self):
            
                    x = tf.keras.layers.Dense(self.out_sz,activation='linear')(self.mdl_imp)
                    x = tf.keras.layers.LayerNormalization(axis=-1)(x)        
                    x = tf.keras.activations.relu(x)                          
                    x = tf.keras.layers.Dropout(0.1)(x)
                    mdl = tf.keras.Model(inputs = self.mdl_imp, outputs=x, name=self.mdl_name)
            
                    return mdl


class Dec_X_Proj(tf.keras.Model):
                    def __init__(self,inp_sz,out_sz,mdl_name,tp, **kwargs):
                              super(Dec_X_Proj, self).__init__(**kwargs)       
                    
                              #Project latent space manifold samples to back original dimensions for expression, methylation and accessibility.
                    
                              self.inp_sz = inp_sz
                              self.out_sz = out_sz
                              self.mdl_name = mdl_name
                              self.tp = tp
                    
                    
                    def call(self):
                    
                    
                              mdl_imp =  tf.keras.Input(shape=(self.inp_sz,))
                              x_dec_proj  = tf.keras.layers.Dense(self.inp_sz,activation='linear')(mdl_imp)
                              x_dec_proj  = tf.keras.layers.LayerNormalization(axis=-1)(x_dec_proj)
                              x_dec_proj  = tf.keras.activations.relu(x_dec_proj)
                              x_dec_proj  = tf.keras.layers.Dropout(0.1)(x_dec_proj)
                              x_dec_proj  = tf.keras.layers.Dense(2*self.inp_sz,activation='linear')(x_dec_proj)
                              x_dec_proj  = tf.keras.layers.LayerNormalization(axis=-1)(x_dec_proj)
                              x_dec_proj  = tf.keras.activations.relu(x_dec_proj)
                              x_dec_proj  = tf.keras.layers.Dropout(0.1)(x_dec_proj)
                    
                    
                              if self.tp == 'tp1':
                    
                                        decoder_outputs_tp1 = tf.keras.layers.Dense(self.out_sz,activation='linear')(x_dec_proj)
                                        decoder_proj = tf.keras.Model(inputs = mdl_imp, outputs = decoder_outputs_tp1, name=self.mdl_name)
                                      
                              elif self.tp == 'tp2':
                    
                                        decoder_outputs_tp2_r = tf.keras.layers.Dense(self.out_sz,activation='linear')(x_dec_proj)
                                        decoder_outputs_tp2_theta = tf.keras.layers.Dense(self.out_sz,activation='linear')(x_dec_proj)
                                        decoder_outputs_tp2_drop = tf.keras.layers.Dense(self.out_sz,activation='linear')(x_dec_proj)
                                        decoder_outputs_tp2 = [decoder_outputs_tp2_r,decoder_outputs_tp2_theta,decoder_outputs_tp2_drop]
                                        decoder_proj = tf.keras.Model(inputs = mdl_imp, outputs = decoder_outputs_tp2, name=self.mdl_name)
                              else:
                                        raise ValueError(
                                                          "tp label accepts two options tp1/tp2"
                                                          )      
                    
                              return decoder_proj

