# STI_mpx
An intercourse-wise agent-based model for mpox and STI outbreak

### Package Versions

- **numpy:** 1.22.4
- **matplotlib:** 3.7.1
- **covasim:** 3.1.4
- **sciris:** 2.1.0
- **pandas:** 1.4.3

### Install require package:
pip install numpy==1.22.4 matplotlib==3.7.1 covasim==3.1.4 sciris==2.1.0 pandas==1.4.3


- for any updates of Covasim package, please refer to https://github.com/InstituteforDiseaseModeling/covasim

### Code Execution Steps
Follow these steps to execute the code and generate the desired results:

1. Run main.py:

Execute the main Python script to initiate the simulation and generate log files.

``python main.py``

2. Generate result_dict.py from log:

Use the log files generated in the previous step to create a result dictionary.

``python log_to_result_dict.py``

3. Run picture.py to generate visualizations:

Execute the script to generate visualizations from the result dictionary.

``python result_dict_to_picture.py``

Please make sure to run these commands in the correct sequence to ensure the code's proper execution. Adjust the paths or commands based on your specific file structure and naming conventions.


