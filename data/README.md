## Dataset Generation via. Sionna and Blender
Due to the scarcity of classified adversarial RF data, the models are trained from scratch using a highly realistic synthetic dataset generated via **Nvidia Sionna RT**. 
* **Environment:** OpenStreetMap data processed into Mitsuba XML via Blender.
* **Propagation Modeling:** Differentiable ray tracing simulating signal strength, path loss, shadowing, diffraction, and multi-path fading. 
* **Data Format:** $3D$ spatial tensors $(X, Y, \text{Frequency})$ stored in Parquet format.

*Average Path Gain Calculation:*
$$g_{i}=\frac{1}{|C_{i}|}\int_{C_{i}}|h(s)|^{2}ds$$

The dataset we used to train our models can be accessed [here](https://huggingface.co/datasets/KR-init/Spectrum-Cartography-256x256-UCF-50K) via. hugging face.