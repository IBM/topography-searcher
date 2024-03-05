# Fill-tuning

We now use the minima of the continous roughness surface to further training our model. We term this 'fill-tuning', as it attempts to fill gaps in the model's knowledge through continued self-supervised pretraining.

We take the embeddings vectors of the top 100 minima and decode them, giving our dataset for fill-tuning. We then load the base model checkpoint and continue training using the same training procedure as the model's original pretraining. For SELFIES-TED-mini, this means we mask 15% of the input tokens, and set the labels as the original SELFIES.

In order to run this step, you must set up the SELFIES-TED code, by following [these intructions](../README.md#selfies-ted)