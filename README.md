# Physics-Informed Neural Network Simulation

The objective of this repository is to model/analyze magnetohydrodynamic (MHD) mixed convection heat transfer inside a lid-driven triangular cavity with a circular obstacle using neural networks within the 3 following cases:


* Case 1: Purely data-driven DNN's (deep neural networks)
* Case 2: Limited data-driven 
* Case 3: Purely physics-informed

## Problem Simplifications
To optimize network execution and streamline loss equations:
1. **Reynolds Number ($\text{Re} = 1$):** We set the non-dimensional Reynolds number to $\text{Re} = 1$. This simplifies the viscous diffusion operator $\frac{1}{\text{Re}}\nabla^2 \mathbf{u} \to \nabla^2 \mathbf{u}$ and eliminates the $\frac{1}{\text{Re}}$ coefficient across all momentum and Lorentz force terms without disrupting viscous drag from the sliding lid.
2. **Prandtl Number ($\text{Pr} = 1$):** Assuming the unit Prandtl number simplifies the thermal diffusion term $\frac{1}{\text{Re}\cdot\text{Pr}}\nabla^2 \theta \to \nabla^2 \theta$.


## Mathematical setup

1. Governing PDEs (Navier Stokes + Energy)

* **Mass Continuity:**
  $$\frac{\partial U}{\partial X} + \frac{\partial V}{\partial Y} = 0$$

* **$X$-Momentum:**
  $$U \frac{\partial U}{\partial X} + V \frac{\partial U}{\partial Y} + \frac{\partial P}{\partial X} - \left( \frac{\partial^2 U}{\partial X^2} + \frac{\partial^2 U}{\partial Y^2} \right) = 0$$

* **$Y$-Momentum (with Buoyancy & Lorentz Drag):**
  $$U \frac{\partial V}{\partial X} + V \frac{\partial V}{\partial Y} + \frac{\partial P}{\partial Y} - \left( \frac{\partial^2 V}{\partial X^2} + \frac{\partial^2 V}{\partial Y^2} \right) - \text{Ri} \, \theta + \text{Ha}^2 \, V = 0$$

* **Energy Conservation:**
  $$U \frac{\partial \theta}{\partial X} + V \frac{\partial \theta}{\partial Y} - \left( \frac{\partial^2 \theta}{\partial X^2} + \frac{\partial^2 \theta}{\partial Y^2} \right) = 0$$

Where:
* $U, V$: Dimensionless velocity components in $X$ and $Y$ directions
* $P$: Dimensionless fluid pressure
* $\theta$: Dimensionless temperature ($\theta = 1$ is Hot, $\theta = 0$ is Cold)
* $\text{Ri}$: Richardson number (Buoyancy vs. Shear forces, $\text{Ri} = \text{Gr}/\text{Re}^2$)
* $\text{Ha}$: Hartmann number (Magnetic force vs. Viscous forces)

2. Boundary Conditions

The domain of interest consists of a sliding top lid, two stationary sidewalls, and an internal circular obstacle:

| Boundary | Velocity ($U, V$) | Temperature ($\theta$) | Physical Meaning |
| :--- | :--- | :--- | :--- |
| **Top Lid** ($Y = Y_{\text{top}}$) | $U = 1, \; V = 0$ | $\theta = 1$ | Moving, Hot Boundary |
| **Left Wall** | $U = 0, \; V = 0$ | $\frac{\partial \theta}{\partial n} = 0$ | No-slip, Adiabatic (Insulated) Boundary |
| **Right Wall** | $U = 0, \; V = 0$ | $\frac{\partial \theta}{\partial n} = 0$ | No-slip, Adiabatic (Insulated) Boundary |
| **Circular Obstacle** | $U = 0, \; V = 0$ | $\theta = 0$ | No-slip, Cold Internal Obstacle |
## Case 1
- Neural networks purely fed data
- Fully supervised learning

## Case 2
- Trained on t in [0,0.4]
- Expected extrapolation error for t > 0.4

## Case 3
- No training data
- Loss:
* $$\mathcal{L}_{\text{total}} = w_{\text{pde}} \mathcal{L}_{\text{PDE}} + w_{\text{bc}} \mathcal{L}_{\text{BC}}$$

* **PDE Residual Loss ($\text{Re} = 1, \text{Pr} = 1$):**
  $$\mathcal{L}_{\text{PDE}} = \frac{1}{N_f} \sum \left( \mathcal{R}_{\text{mass}}^2 + \mathcal{R}_{u\text{-mom}}^2 + \mathcal{R}_{v\text{-mom}}^2 + \mathcal{R}_{\text{energy}}^2 \right)$$

* **Mass Continuity Residual:**
  $$\mathcal{R}_{\text{mass}} = \frac{\partial U}{\partial X} + \frac{\partial V}{\partial Y}$$

* **$X$-Momentum Residual:**
  $$\mathcal{R}_{u\text{-mom}} = U \frac{\partial U}{\partial X} + V \frac{\partial U}{\partial Y} + \frac{\partial P}{\partial X} - \left( \frac{\partial^2 U}{\partial X^2} + \frac{\partial^2 U}{\partial Y^2} \right)$$

* **$Y$-Momentum Residual:**
  $$\mathcal{R}_{v\text{-mom}} = U \frac{\partial V}{\partial X} + V \frac{\partial V}{\partial Y} + \frac{\partial P}{\partial Y} - \left( \frac{\partial^2 V}{\partial X^2} + \frac{\partial^2 V}{\partial Y^2} \right) - \text{Ri} \, \theta + \text{Ha}^2 \, V$$

* **Energy Conservation Residual:**
  $$\mathcal{R}_{\text{energy}} = U \frac{\partial \theta}{\partial X} + V \frac{\partial \theta}{\partial Y} - \left( \frac{\partial^2 \theta}{\partial X^2} + \frac{\partial^2 \theta}{\partial Y^2} \right)$$

* **Boundary Condition Loss:**
  $$\mathcal{L}_{\text{BC}} = \mathcal{L}_{\text{vel}} + \mathcal{L}_{\theta,\text{top}} + \mathcal{L}_{\theta,\text{obstacle}} + \mathcal{L}_{\theta,\text{sidewalls}}$$

  * **Velocity Loss (All Solid Walls & Top Lid):**
    $$\mathcal{L}_{\text{vel}} = \frac{1}{N_b} \sum \left( (U - U_{\text{wall}})^2 + (V - V_{\text{wall}})^2 \right)$$

  * **Top Lid Thermal Loss ($\theta = 1$):**
    $$\mathcal{L}_{\theta,\text{top}} = \frac{1}{N_{\text{top}}} \sum \left( \theta - 1 \right)^2$$

  * **Cold Obstacle Thermal Loss ($\theta = 0$):**
    $$\mathcal{L}_{\theta,\text{obstacle}} = \frac{1}{N_{\text{obstacle}}} \sum \left( \theta - 0 \right)^2$$

  * **Adiabatic Sidewalls Thermal Loss ($\frac{\partial \theta}{\partial n} = 0$):**
    $$\mathcal{L}_{\theta,\text{sidewalls}} = \frac{1}{N_{\text{side}}} \sum \left( n_x \frac{\partial \theta}{\partial X} + n_y \frac{\partial \theta}{\partial Y} \right)^2$$
## Key Observations
- Physics-informed models are only fed governing equations, no data
- Data-driven are only given data points, no physics
- Physics-informed predictive models are more accurate than data-driven models

## Author
Ameera Junaid