# README: Hybrid Neural Networks in the Mushroom Body Drive Olfactory Preference in Drosophila
# Project Overview
We provide the code used to reproduce the results presented in our paper, “Hybrid Neural Networks in the Mushroom Body Drive Olfactory Preference in Drosophila.”

Our code performs a range of connectomic analyses using data from the hemibrain and FAFB datasets, and further integrates functional data from the DoOR database and behavioral data from Knaden et al., 2012.

Due to data size restriction, we have archived the data in: https://doi.org/10.5281/zenodo.15263535. 
By unzip the files from zenodo, please execute main.py.

# References:
Behavioral Data:
Knaden, M., Strutz, A., Ahsan, J., Sachse, S., & Hansson, B. S. (2012). Spatial representation of odorant valence in an insect brain. Cell reports, 1(4), 392-399.
Functional Data:
Münch, D., & Galizia, C. G. (2016). DoOR 2.0-comprehensive mapping of Drosophila melanogaster odorant responses. Scientific reports, 6(1), 21841.
Connectomic Data:
Scheffer, L. K., Xu, C. S., Januszewski, M., Lu, Z., Takemura, S. Y., Hayworth, K. J., … & Plaza, S. M. (2020). A connectome and analysis of the adult Drosophila central brain. elife, 9, e57443.
Zheng, Z., Li, F., Fisher, C., Ali, I. J., Sharifi, N., Calle-Schuler, S., … & Bock, D. D. (2022). Structured sampling of olfactory input by the fly mushroom body. Current Biology, 32(15), 3334-3349.
Zheng, Z., Lauritzen, J. S., Perlman, E., Robinson, C. G., Nichols, M., Milkie, D., … & Bock, D. D. (2018). A complete electron microscopy volume of the brain of adult Drosophila melanogaster. Cell, 174(3), 730-743.

# Files and variables
### main.py\
Executes the core analyses, including spatial distribution analysis, behavioral analysis, and simulations. Each figure is annotated within the corresponding function.
### PN_to_KC_coding_simulation.py\
Performs simulations of artificial odors for Figure 4.
### Analyze_result.py\
Analyzes simulation results, focusing on acuity and dimensionality.
### simulation_process.py\
Generates artificial odors used in the simulations.
### behavioral_analysis.py\
Analyzes the correlation between connection preferences and behavioral biases.
### generate_connection.py\
Explores different PN-to-KC wiring configurations.
### MGPN_analysis.py\
Analyzes multi-glomerular PN-to-KC connections.
### read_DoOR.py\
Preprocesses data from the DoOR (Database of Odorant Responses).
### extract_bouton_claw.py\
Extracts PN boutons and KC claws from anatomical data.
### function_data_processing.py\
Analyzes functional imaging results.
### analysis_tool.py\
Contains utility functions for analyzing spatial distribution data.
### shuffling_20241117.ipynb\
Jupyter notebook for functional predictions and connection preference analysis.Access information
Other publicly accessible locations of the data:

Brain Research Center, National Tsing Hua University, Taiwan
Data was derived from the following sources:

Brain Research Center, National Tsing Hua University, Taiwan
