import numpy as np

# The original ten equation-native formulas below are copied verbatim.

FUNCTIONS = {
    "exp_decay": (
        1,
        lambda x: 1.7 * np.exp(-0.8 * x[:, 0]) + 0.25,
        "1.7*exp(-0.8*x1)+0.25",
    ),
    "log_response": (
        1,
        lambda x: 1.2 * np.log(1.5 * x[:, 0] + 0.7) - 0.15,
        "1.2*log(1.5*x1+0.7)-0.15",
    ),
    "power_law": (
        1,
        lambda x: 0.8 * np.exp(1.35 * np.log(x[:, 0] + 0.2)) + 0.1,
        "0.8*exp(1.35*log(x1+0.2))+0.1",
    ),
    "arrhenius_like": (
        1,
        lambda x: 2.0 * np.exp(-1.15 / (x[:, 0] + 0.6)),
        "2.0*exp(-1.15/(x1+0.6))",
    ),
    "michaelis_menten": (
        1,
        lambda x: 1.8 * x[:, 0] / (0.45 + x[:, 0]),
        "1.8*x1/(0.45+x1)",
    ),
    "multiplicative_power": (
        2,
        lambda x: 1.1 * np.exp(0.7 * np.log(x[:, 0] + 0.15) - 0.45 * np.log(x[:, 1] + 0.25)),
        "1.1*exp(0.7*log(x1+0.15)-0.45*log(x2+0.25))",
    ),
    "mixed_exp_log": (
        1,
        lambda x: 0.9 * np.exp(0.55 * x[:, 0]) + 0.65 * np.log(1.2 * x[:, 0] + 0.8),
        "0.9*exp(0.55*x1)+0.65*log(1.2*x1+0.8)",
    ),
    "damped_log_response": (
        1,
        lambda x: 1.4 * np.exp(-0.55 * x[:, 0]) * np.log(1.7 * x[:, 0] + 0.9),
        "1.4*exp(-0.55*x1)*log(1.7*x1+0.9)",
    ),
    "exp_saturation": (
        1,
        lambda x: 1.3 * (1.0 - np.exp(-1.1 * x[:, 0])) + 0.05,
        "1.3*(1-exp(-1.1*x1))+0.05",
    ),
    "two_var_decay": (
        2,
        lambda x: 0.75 * x[:, 0] * np.exp(-0.65 * x[:, 1]) + 0.2,
        "0.75*x1*exp(-0.65*x2)+0.2",
    ),
}


# Four prespecified out-of-family controls, newly added in this follow-up.
ORIGINAL_DATASETS = tuple(FUNCTIONS)
FUNCTIONS.update({
    "control_quadratic": (1, lambda x: x[:,0]**2+x[:,0]+1, "x1**2+x1+1"),
    "control_kinetic_energy": (2, lambda x: 0.5*x[:,0]*x[:,1]**2, "0.5*x1*x2**2"),
    "control_sine": (1, lambda x: np.sin(2*x[:,0]), "sin(2*x1)"),
    "control_sine_cosine": (2, lambda x: np.sin(1.5*x[:,0])+0.3*np.cos(2*x[:,1]), "sin(1.5*x1)+0.3*cos(2*x2)"),
})
