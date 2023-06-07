#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install h2o


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().system('apt-get install default-jre')
get_ipython().system('java -version')


# In[ ]:


get_ipython().system('pip install h2o')


# In[ ]:


import h2o


# In[ ]:


h2o.init()


# In[ ]:


from h2o.automl import H2OAutoML


# In[ ]:


df = h2o.import_file('/content/drive/MyDrive/IIIT-K Intern/country_wise_latest.csv')


# In[ ]:


df.types


# In[ ]:


df.describe()


# In[ ]:


df_train,df_test,df_valid = df.split_frame(ratios=[.7, .15])


# In[ ]:


df_train


# In[ ]:


y = "WHO Region"
x = df.columns
x.remove(y)
x.remove("Country/Region")


# In[ ]:


aml = H2OAutoML(max_models = 10, seed = 10, exclude_algos = ["StackedEnsemble", "DeepLearning"], verbosity="info", nfolds=0)


# In[ ]:


aml.train(x = x, y = y, training_frame = df_train, validation_frame=df_valid)


# In[ ]:


lb = aml.leaderboard


# In[ ]:


lb.head()


# In[ ]:


df_pred=aml.leader.predict(df_test)


# In[ ]:


df_pred.head()


# In[ ]:


aml.leader.model_performance(df_test)


# In[ ]:


model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])


# In[ ]:


model_ids


# In[ ]:


h2o.get_model([mid for mid in model_ids if "XGBoost" in mid][0])


# In[ ]:


out = h2o.get_model([mid for mid in model_ids if "XGBoost" in mid][0])


# In[ ]:


out.params


# In[ ]:


out.convert_H2OXGBoostParams_2_XGBoostParams()


# In[ ]:


out


# In[ ]:


out_gbm = h2o.get_model([mid for mid in model_ids if "GBM" in mid][0])


# In[ ]:


out.confusion_matrix()


# In[ ]:


out.varimp_plot()


# In[ ]:




