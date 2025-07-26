# PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks

## Publication

Implementation of the paper "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks."

Authors: Leo Zhiyuan Zhao, Xueying Ding, B.Aditya Prakash

Placement: ICLR 2024 Poster

Paper + Appendix: [https://arxiv.org/abs/2307.11833](https://arxiv.org/abs/2307.11833)

## Training

We also provide demo notebooks for convection, 1d_reaction, 1d_wave, and Navier-Stokes PDEs. The demos include all code for training, testing, and ground truth acquirement.

To visualize the loss landscape, run the above command to train and save the model first, then run the script:

```
python3 vis_landscape.py
```

Please adapt the model path accordingly.

## Contact

If you have any questions about the code, please contact Leo Zhiyuan Zhao at  ```leozhao1997[at]gatech[dot]edu```.

## Citation

If you find our work useful, please cite our work:

```
@article{zhao2023pinnsformer,
  title={PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks},
  author={Zhao, Leo Zhiyuan and Ding, Xueying and Prakash, B Aditya},
  journal={arXiv preprint arXiv:2307.11833},
  year={2023}
}
```
## AI Agentic Systems Epidemiology Extension (AI Safety x Physics)

**New Components:**
- **`ai_epidemiology_model.py`**: SEIR epidemiological model for AI agent security, with PINN solver
- **`bifurcation_analysis.py`**: Bifurcation and stability analysis for intervention strategies
- **`demo/ai_epidemiology/`**: Jupyter notebook demonstrating the model

**AI Agent SEIR Model:**
- **S(t)**: Susceptible agents (vulnerable to attacks)
- **E(t)**: Exposed agents (compromised but not spreading)
- **I(t)**: Infected agents (exhibiting malignant behavior)
- **R(t)**: Removed agents (isolated/patched/immunized)

**Key Parameters:**
- **β**: Attack transmission rate (depends on ASR)
- **σ**: Incubation rate (exposed → infected)
- **γ**: Detection/isolation rate
- **ν**: Immunization/patching rate
- **α**: External attack pressure

**Applications:**
- Model attack spread in multi-agent AI systems
- Analyze intervention strategies and critical thresholds
- Identify bifurcation points for early warning systems
- Optimize resource allocation for cybersecurity

**Run AI epidemiology model analysis:**
```bash
python3 ai_epidemiology_model.py
```

**Run bifurcation analysis:**
```bash
python3 bifurcation_analysis.py
```